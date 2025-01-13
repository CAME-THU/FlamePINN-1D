"""Case configuration: calculation domain, parameters, reference solutions, governing equations, IC/BC/OCs."""
# import math as mt
import numpy as np
import cantera as ct
import torch
import deepxde as dde
from scipy.interpolate import interp1d
from utils.icbcs import ScaledDirichletBC, ScaledPointSetBC
from utils.gas1d import Gas1D_1stepIr_Ea

# dtype = dde.config.real(torch)
dtype = torch.float64

# R = ct.gas_constant
R = 8314.46261815324  # J/kmol/K


class Case:
    def __init__(self, args):
        self.args = args

        # ----------------------------------------------------------------------
        # define calculation domain
        self.x_l, self.x_r = 0.0, args.length
        self.geom = dde.geometry.Interval(self.x_l, self.x_r)

        # ----------------------------------------------------------------------
        # define the names of independents, dependents, and equations
        n_spe = args.gas.n_species
        self.names = {
            "independents": ["x"],
            "dependents": ["u", "V", "T"] + ["Y_" + args.gas.species_names[k] for k in range(n_spe)],
            "equations": ["continuity", "momentum_r", "energy"]
                         + ["specie_" + args.gas.species_names[k] for k in range(n_spe)],
            "ICBCOCs": []}

        self.icbcocs = []  # initial, boundary, observation conditions

        self.pCurv_infe_s, self.A_infe_s, self.Ea_infe_s = None, None, None
        if "pCurv" in args.infer_paras:
            self.pCurv_infe_s = dde.Variable(args.infer_paras["pCurv"] * args.scales["pCurv"], dtype=dtype)
        if "Ea" in args.infer_paras:
            self.Ea_infe_s = dde.Variable(args.infer_paras["Ea"] * args.scales["Ea"], dtype=dtype)

        if "Ea" in args.infer_paras:
            self.gas1d = Gas1D_1stepIr_Ea(args.gas, (self.Ea_infe_s, args.scales["Ea"]))
        else:
            self.gas1d = args.gas1d

        self.Ea = self.gas1d.AbEs[:, 2].cpu().numpy().item()

        # ----------------------------------------------------------------------
        # define parameters
        self.gas = args.gas
        self.p = args.p
        self.T_u = args.T_u
        self.T_b = args.T_b
        self.mdot_u = args.mdot_u
        self.mdot_b = args.mdot_b
        self.phi = args.phi
        self.fuel, self.oxidizer = args.fuel, args.oxidizer

        args.gas.X = "N2:1"
        self.Ys_b = args.gas.Y
        self.Xs_b = args.gas.X

        self.gas.set_equivalence_ratio(self.phi, self.fuel, self.oxidizer)
        self.Ys_u = self.gas.Y
        self.Xs_u = self.gas.X

        W_u = 1. / float(np.sum(self.Ys_u / self.gas.molecular_weights))  # kg/kmol
        W_b = 1. / float(np.sum(self.Ys_b / self.gas.molecular_weights))  # kg/kmol
        rho_u = self.p * W_u / (R * self.T_u)  # kg/m3
        rho_b = self.p * W_b / (R * self.T_b)  # kg/m3
        self.u_u = self.mdot_u / rho_u
        self.u_b = -self.mdot_b / rho_b

        # ----------------------------------------------------------------------
        # load data
        if args.case_id == 1:
            load_name = "1S_CH4_MP1"
        else:
            load_name = None
            # TODO: more cases.
        # load_dir = f"../ref_solution/results/{case_name}/data/"
        load_dir = "../ref_solution/results/{:s}/p{:.2f}_phi{:.2f}/data/".format(load_name, self.p / 101325, self.phi)
        x_src = np.load(load_dir + "x.npy")
        u_src = np.load(load_dir + "u.npy")
        V_src = np.load(load_dir + "V.npy")
        T_src = np.load(load_dir + "T.npy")
        Ys_src = np.load(load_dir + "Ys.npy")
        rho_src = np.load(load_dir + "rho.npy")
        self.pCurv_refe = np.load(load_dir + "pCurv.npy").item()

        # ----------------------------------------------------------------------
        # construct interpolation function
        self.func_u_interp = interp1d(x_src, u_src, kind="linear")
        self.func_V_interp = interp1d(x_src, V_src, kind="linear")
        self.func_T_interp = interp1d(x_src, T_src, kind="linear")
        self.func_Ys_interp = interp1d(x_src, Ys_src.T, kind="linear")
        self.func_rho_interp = interp1d(x_src, rho_src, kind="linear", fill_value="extrapolate")

        # ----------------------------------------------------------------------
        # define ICs, BCs, OCs
        self.define_icbcocs()

    # ----------------------------------------------------------------------
    # reference solution with individual inputs
    def func_u(self, x):
        return self.func_u_interp(x)

    def func_V(self, x):
        return self.func_V_interp(x)

    def func_T(self, x):
        return self.func_T_interp(x)

    def func_Ys(self, x):
        return self.func_Ys_interp(x)[:, :, 0].T

    def func_rho(self, x):
        return self.func_rho_interp(x)

    # ----------------------------------------------------------------------
    # define ode
    def ode(self, x, uVTYs):
        args = self.args
        scale_u, scale_V, scale_T = args.scales["u"], args.scales["V"], args.scales["T"]
        scale_Ys = torch.tensor(args.scales["Ys"], dtype=dtype)  # (n_spe, )
        scale_x = args.scales["x"]
        # shift_x = args.shifts["x"]
        # gas1d = args.gas1d
        gas1d = self.gas1d

        u, V, T, Ys = uVTYs[:, 0:1], uVTYs[:, 1:2], uVTYs[:, 2:3], uVTYs[:, 3:]

        u_x = dde.grad.jacobian(uVTYs, x, i=0, j=0)
        V_x = dde.grad.jacobian(uVTYs, x, i=1, j=0)
        V_xx = dde.grad.hessian(uVTYs, x, component=1, i=0, j=0)
        T_x = dde.grad.jacobian(uVTYs, x, i=2, j=0)
        T_xx = dde.grad.hessian(uVTYs, x, component=2, i=0, j=0)
        Ys_x = torch.hstack([dde.grad.jacobian(uVTYs, x, i=k + 3, j=0) for k in range(gas1d.n_spe)])  # (n_pnt, n_spe)

        rho = gas1d.cal_rho(T, Ys)  # (n_pnt, 1)
        cps = gas1d.cal_cps(T)  # (n_pnt, n_spe)
        cp = gas1d.cal_cp(T, Ys)  # (n_pnt, 1)
        mu = gas1d.cal_mu(T, Ys)  # (n_pnt, 1)
        lam = gas1d.cal_lam(T, Ys)  # (n_pnt, 1)
        Dkm_apo = gas1d.cal_Dkms_apo(T, Ys)  # (n_pnt, n_spe)
        W = gas1d.cal_W(Ys)  # (n_pnt, 1)
        Xs = gas1d.cal_Xs(Ys)  # (n_pnt, n_spe)
        wdot = gas1d.cal_omega_dot_mass(T, Ys)  # (n_pnt, n_spe)
        wTdot = gas1d.cal_omegaT_dot(T, Ys)  # (n_pnt, 1)

        rho_x = dde.grad.jacobian(rho, x, i=0, j=0)
        mu_x = dde.grad.jacobian(mu, x, i=0, j=0)
        lam_x = dde.grad.jacobian(lam, x, i=0, j=0)
        Xs_x = torch.hstack([dde.grad.jacobian(Xs, x, i=k, j=0) for k in range(gas1d.n_spe)])  # (n_pnt, n_spe)
        js = -rho * (gas1d.Ws / W) * Dkm_apo * Xs_x  # (n_pnt, n_spe)
        js = js - Ys * torch.sum(js, dim=-1, keepdim=True)  # (n_pnt, n_spe)
        js_x = torch.hstack([dde.grad.jacobian(js, x, i=k, j=0) for k in range(gas1d.n_spe)])  # (n_pnt, n_spe)
        j_cp_sum = torch.sum(js * cps, dim=-1, keepdim=True)  # (n_pnt, 1)

        pCurv = self.pCurv_infe_s / args.scales["pCurv"] if "pCurv" in args.infer_paras else self.pCurv_refe

        continuity = (rho * u_x + u * rho_x) + 2 * rho * V
        momentum_r = rho * u * V_x + rho * V ** 2 + pCurv - (mu * V_xx + mu_x * V_x)
        energy = rho * u * cp * T_x - (lam * T_xx + lam_x * T_x) + j_cp_sum * T_x - wTdot
        species = rho * u * Ys_x + js_x - wdot

        continuity *= scale_u / scale_x
        momentum_r *= scale_u * scale_V / scale_x
        energy *= scale_u * scale_T / scale_x / 1000
        species *= scale_u * scale_Ys / scale_x
        return [continuity, momentum_r, energy] + [species[:, k: k + 1] for k in range(gas1d.n_spe)]

    # ----------------------------------------------------------------------
    # define ICs, BCs, OCs
    def define_icbcocs(self):
        args = self.args
        n_spe = args.gas.n_species
        # geom = self.geom
        x_l, x_r = self.x_l, self.x_r
        scale_u, scale_V, scale_T = args.scales["u"], args.scales["V"], args.scales["T"]
        scale_Ys = args.scales["Ys"]

        bdr_x = np.array([x_l, x_r])[:, None]
        bdr_u = np.array([self.u_u, self.u_b])[:, None]
        bdr_V = np.array([0.0, 0.0])[:, None]
        bdr_T = np.array([self.T_u, self.T_b])[:, None]
        bdr_Ys = np.vstack([self.Ys_u, self.Ys_b])

        bc_u = ScaledPointSetBC(bdr_x, bdr_u, component=0, scale=scale_u)
        bc_V = ScaledPointSetBC(bdr_x, bdr_V, component=1, scale=scale_V)
        bc_T = ScaledPointSetBC(bdr_x, bdr_T, component=2, scale=scale_T)
        bcs_Ys = [ScaledPointSetBC(bdr_x, bdr_Ys[:, k:k + 1], component=k + 3, scale=scale_Ys[k])
                  for k in range(n_spe)]

        if args.bc_type == "soft":
            self.icbcocs += [bc_u, bc_V, bc_T] + bcs_Ys
            self.names["ICBCOCs"] += (["BC_u", "BC_V", "BC_T"]
                                      + ["BC_" + args.gas.species_names[k] for k in range(n_spe)])
        else:  # "none"
            pass

        if args.oc_type == "soft":
            n_ob = args.n_ob
            ob_x = np.linspace(x_l, x_r, n_ob)[:, None]

            ob_u = self.func_u(ob_x)
            ob_V = self.func_V(ob_x)
            ob_T = self.func_T(ob_x)
            ob_Ys = self.func_Ys(ob_x)  # (n_ob, n_spe)

            # change scales
            args.scales["u"] = 5.0 / np.max(np.abs(ob_u))
            args.scales["V"] = 5.0 / np.max(ob_V)
            args.scales["T"] = 5.0 / np.max(ob_T)
            scale_Ys = 5.0 / np.max(ob_Ys, axis=0)
            args.scales["Ys"] = scale_Ys.tolist()
            # self.args = args

            normal_noise_u = np.random.randn(len(ob_u))[:, None]
            normal_noise_V = np.random.randn(len(ob_V))[:, None]
            normal_noise_T = np.random.randn(len(ob_T))[:, None]
            normal_noise_Ys = np.random.randn(ob_Ys.size).reshape([ob_Ys.shape[0], ob_Ys.shape[1]])
            ob_u += normal_noise_u * ob_u * args.noise_level
            ob_V += normal_noise_V * ob_V * args.noise_level
            ob_T += normal_noise_T * ob_T * args.noise_level
            ob_Ys += normal_noise_Ys * ob_Ys * args.noise_level

            oc_u = ScaledPointSetBC(ob_x, ob_u, component=0, scale=args.scales["u"])
            oc_V = ScaledPointSetBC(ob_x, ob_V, component=1, scale=args.scales["V"])
            oc_T = ScaledPointSetBC(ob_x, ob_T, component=2, scale=args.scales["T"])
            ocs_Ys = [ScaledPointSetBC(ob_x, ob_Ys[:, k:k + 1], component=k + 3, scale=args.scales["Ys"][k])
                      for k in range(n_spe)]
            self.icbcocs += [oc_u, oc_V, oc_T] + ocs_Ys
            self.names["ICBCOCs"] += (["OC_u", "OC_V", "OC_T"]
                                      + ["OC_" + args.gas.species_names[k] for k in range(n_spe)])
        else:  # "none"
            n_ob = 0
            ob_x = np.empty([1, 1])
            ob_u, ob_V, ob_T, ob_Ys = np.empty([1, 1]), np.empty([1, 1]), np.empty([1, 1]), np.empty([1, n_spe])
        self.n_ob = n_ob
        self.ob_x = ob_x
        self.ob_u, self.ob_V, self.ob_T, self.ob_Ys = ob_u, ob_V, ob_T, ob_Ys
