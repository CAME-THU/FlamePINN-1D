"""Case configuration: calculation domain, parameters, reference solutions, governing equations, IC/BC/OCs."""
# import math as mt
import numpy as np
import cantera as ct
import torch
import deepxde as dde
from scipy.interpolate import interp1d
from utils.icbcs import ScaledDirichletBC, ScaledPointSetBC

# dtype = dde.config.real(torch)
dtype = torch.float64

# R = ct.gas_constant


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
            "dependents": ["T"] + ["Y_" + args.gas.species_names[k] for k in range(n_spe)],
            "equations": ["energy"] + ["specie_" + args.gas.species_names[k] for k in range(n_spe)],
            "ICBCOCs": []}

        self.icbcocs = []  # initial, boundary, observation conditions

        self.sL_infe_s = dde.Variable(args.infer_paras["sL"] * args.scales["sL"], dtype=dtype) if "sL" in args.infer_paras else None
        
        # ----------------------------------------------------------------------
        # define parameters
        self.gas = args.gas
        self.p = args.p
        self.T_in = args.T_in

        self.phi = args.phi
        self.fuel, self.oxidizer = args.fuel, args.oxidizer

        self.gas.TP = self.T_in, self.p
        self.gas.set_equivalence_ratio(self.phi, self.fuel, self.oxidizer)
        self.Ys_in = self.gas.Y
        self.Xs_in = self.gas.X
        self.rho_in = self.gas.density

        # initial guess from Cantera
        sim = ct.FreeFlame(self.gas, grid=np.linspace(self.x_l, self.x_r, 501))
        sim.boundary_emissivities = 0.0, 0.0
        sim.radiation_enabled = False
        sim.set_initial_guess()
        self.fix_x = sim.fixed_temperature_location
        self.fix_T = sim.fixed_temperature

        self.gas.equilibrate("HP")  # adiabatic
        self.T_eq = self.gas.T
        self.Xs_eq = self.gas.X
        self.Ys_eq = self.gas.Y

        # change scale_Ys
        args.scales["T"] = 5.0 / self.T_eq
        scale_Ys = 5.0 / np.maximum(self.Ys_in, self.Ys_eq)
        args.scales["Ys"] = scale_Ys.tolist()
        # self.args = args

        # ----------------------------------------------------------------------
        # load data
        if args.case_id == 1:
            load_name = "1S_CH4_MP1"
        else:
            load_name = None
            # TODO: more cases.
        # load_dir = f"../ref_solution/results/{case_name}/data/"
        load_dir = "../ref_solution/results/{:s}/p{:.2f}_phi{:.2f}/data/".format(load_name, self.p/101325, self.phi)
        x_src = np.load(load_dir + "x.npy")
        T_src = np.load(load_dir + "T.npy")
        Ys_src = np.load(load_dir + "Ys.npy")
        u_src = np.load(load_dir + "u.npy")
        rho_src = np.load(load_dir + "rho.npy")
        self.sL_refe = np.load(load_dir + "sL.npy").item()

        # ----------------------------------------------------------------------
        # construct interpolation function
        self.func_T_interp = interp1d(x_src, T_src, kind="linear", fill_value="extrapolate")
        self.func_Ys_interp = interp1d(x_src, Ys_src.T, kind="linear", fill_value="extrapolate")
        self.func_u_interp = interp1d(x_src, u_src, kind="linear", fill_value="extrapolate")
        self.func_rho_interp = interp1d(x_src, rho_src, kind="linear", fill_value="extrapolate")

        # ----------------------------------------------------------------------
        # define ICs, BCs, OCs
        self.define_icbcocs()
        print(args)

    # ----------------------------------------------------------------------
    # reference solution with individual inputs
    def func_T(self, x):
        return self.func_T_interp(x)

    def func_Ys(self, x):
        return self.func_Ys_interp(x)[:, :, 0].T

    def func_u(self, x):
        return self.func_u_interp(x)

    def func_rho(self, x):
        return self.func_rho_interp(x)

    # ----------------------------------------------------------------------
    # initial field for warmup pretraining
    def func_ini_guess(self, x):
        n_spe = self.args.gas.n_species
        result_T = np.zeros_like(x)
        result_Ys = np.zeros([n_spe, len(x)])
        x0, x3 = x[0], x[-1]
        x1, x2 = x[0] + 0.3 * (x[-1] - x[0]), x[0] + 0.5 * (x[-1] - x[0])
        flag1 = np.logical_and(x >= x0, x <= x1)
        flag2 = np.logical_and(x > x1, x <= x2)
        flag3 = np.logical_and(x > x2, x <= x3)

        result_T[flag1] = self.T_in
        result_T[flag3] = self.T_eq
        result_T[flag2] = self.T_in + (self.T_eq - self.T_in) / (x2 - x1) * (x[flag2] - x1)

        for k in range(n_spe):
            result_Ys[k, flag1] = self.Ys_in[k]
            result_Ys[k, flag3] = self.Ys_eq[k]
            result_Ys[k, flag2] = self.Ys_in[k] + (self.Ys_eq[k] - self.Ys_in[k]) / (x2 - x1) * (x[flag2] - x1)

        return np.hstack([result_T[:, None], result_Ys.T])

    # ----------------------------------------------------------------------
    # define ode
    def ode(self, x, TYs):
        args = self.args
        scale_T = args.scales["T"]
        scale_Ys = torch.tensor(args.scales["Ys"], dtype=dtype)  # (n_spe, )
        scale_x = args.scales["x"]
        # shift_x = args.shifts["x"]
        gas1d = args.gas1d

        T, Ys = TYs[:, 0:1], TYs[:, 1:]

        T_x = dde.grad.jacobian(TYs, x, i=0, j=0)
        T_xx = dde.grad.hessian(TYs, x, component=0, i=0, j=0)
        Ys_x = torch.hstack([dde.grad.jacobian(TYs, x, i=k + 1, j=0) for k in range(gas1d.n_spe)])  # (n_pnt, n_spe)

        rho = gas1d.cal_rho(T, Ys)  # (n_pnt, 1)
        cps = gas1d.cal_cps(T)  # (n_pnt, n_spe)
        cp = gas1d.cal_cp(T, Ys)  # (n_pnt, 1)
        # mu = gas1d.cal_mu(T, Ys)  # (n_pnt, 1)
        lam = gas1d.cal_lam(T, Ys)  # (n_pnt, 1)
        Dkm_apo = gas1d.cal_Dkms_apo(T, Ys)  # (n_pnt, n_spe)
        W = gas1d.cal_W(Ys)  # (n_pnt, 1)
        Xs = gas1d.cal_Xs(Ys)  # (n_pnt, n_spe)
        wdot = gas1d.cal_omega_dot_mass(T, Ys)  # (n_pnt, n_spe)
        wTdot = gas1d.cal_omegaT_dot(T, Ys)  # (n_pnt, 1)

        # rho_x = dde.grad.jacobian(rho, x, i=0, j=0)
        # mu_x = dde.grad.jacobian(mu, x, i=0, j=0)
        lam_x = dde.grad.jacobian(lam, x, i=0, j=0)
        Xs_x = torch.hstack([dde.grad.jacobian(Xs, x, i=k, j=0) for k in range(gas1d.n_spe)])  # (n_pnt, n_spe)
        js = -rho * (gas1d.Ws / W) * Dkm_apo * Xs_x  # (n_pnt, n_spe)
        js = js - Ys * torch.sum(js, dim=-1, keepdim=True)  # (n_pnt, n_spe)
        js_x = torch.hstack([dde.grad.jacobian(js, x, i=k, j=0) for k in range(gas1d.n_spe)])  # (n_pnt, n_spe)
        j_cp_sum = torch.sum(js * cps, dim=-1, keepdim=True)  # (n_pnt, 1)

        sL = self.sL_infe_s / args.scales["sL"] if "sL" in args.infer_paras else self.sL_refe

        energy = self.rho_in * sL * cp * T_x - (lam * T_xx + lam_x * T_x) + j_cp_sum * T_x - wTdot
        species = self.rho_in * sL * Ys_x + js_x - wdot

        energy *= scale_T / scale_x / 1000  # 1000 for cp
        species *= scale_Ys / scale_x
        return [energy] + [species[:, k: k + 1] for k in range(gas1d.n_spe)]

    # ----------------------------------------------------------------------
    # define ICs, BCs, OCs
    def define_icbcocs(self):
        args = self.args
        n_spe = args.gas.n_species
        # geom = self.geom
        x_l, x_r = self.x_l, self.x_r
        scale_T = args.scales["T"]
        scale_Ys = args.scales["Ys"]
        # scale_Ys = torch.tensor(args.scales["Ys"], dtype=dtype)  # (n_spe, )

        bdr_x = np.array([x_l, x_r])[:, None]
        bdr_T = np.array([self.T_in, self.T_eq])[:, None]
        bdr_Ys = np.vstack([self.Ys_in, self.Ys_eq])

        bc_T = ScaledPointSetBC(bdr_x, bdr_T, component=0, scale=scale_T)
        bcs_Ys = [ScaledPointSetBC(bdr_x, bdr_Ys[:, k:k+1], component=k+1, scale=scale_Ys[k]) for k in range(n_spe)]

        # fix_x, fix_T = np.array([[x_l + 0.35 * (x_r - x_l)]]), np.array([[0.75 * self.T_in + 0.25 * self.T_eq]])
        fix_x, fix_T = np.array([[self.fix_x]]), np.array([[self.fix_T]])
        fixc_T = ScaledPointSetBC(fix_x, fix_T, component=0, scale=scale_T)
    
        if args.bc_type == "soft":
            self.icbcocs += [bc_T, fixc_T] + bcs_Ys
            self.names["ICBCOCs"] += ["BC_T", "fixC_T"] + ["BC_" + args.gas.species_names[k] for k in range(n_spe)]
        elif args.bc_type == "hard":
            self.icbcocs += [fixc_T]
            self.names["ICBCOCs"] += ["fixC_T"]
        else:  # "none"
            pass

        if args.oc_type == "soft":
            n_ob = 101
            # n_ob = 501
            ob_x = np.linspace(x_l, x_r, n_ob)[:, None]
    
            ob_T = self.func_T(ob_x)
            ob_Ys = self.func_Ys(ob_x)

            normal_noise_T = np.random.randn(len(ob_T))[:, None]
            normal_noise_Ys = np.random.randn(ob_Ys.size).reshape([ob_Ys.shape[0], ob_Ys.shape[1]])
            ob_T += normal_noise_T * ob_T * args.noise_level
            ob_Ys += normal_noise_Ys * ob_Ys * args.noise_level

            oc_T = ScaledPointSetBC(ob_x, ob_T, component=0, scale=scale_T)
            ocs_Ys = [ScaledPointSetBC(ob_x, ob_Ys[:, k:k+1], component=k+1, scale=scale_Ys[k]) for k in range(n_spe)]
            self.icbcocs += [oc_T] + ocs_Ys
            self.names["ICBCOCs"] += ["OC_T"] + ["OC_" + args.gas.species_names[k] for k in range(n_spe)]
        else:  # "none"
            n_ob = 0
            ob_x = np.empty([1, 1])
            ob_T, ob_Ys = np.empty([1, 1]), np.empty([1, n_spe])
        self.n_ob = n_ob

