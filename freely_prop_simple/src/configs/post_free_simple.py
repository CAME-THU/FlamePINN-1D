"""Define the field variables to be post-processed."""
import numpy as np
from utils.postprocess import PostProcess1D


class Postprocess(PostProcess1D):
    def __init__(self, args, case, model, output_dir):
        super().__init__(args, case, model, output_dir)

        # ----------------------------------------------------------------------
        # Get the predicted and reference fields
        output = model.predict(self.x[:, None])
        self.T_pred = output[:, 0]
        self.T_pred = self.T_pred.astype(np.float64)

        W, cp, qF = case.W, case.cp, case.qF
        # R, A, Ea, nu_rxn = case.R, case.A, case.Ea, case.nu_rxn
        R, A, nu_rxn = case.R, case.A, case.nu_rxn
        T_in, YF_in, rho_in = case.T_in, case.YF_in, case.rho_in
        Rg = R / W

        sL = case.sL_infe_s.cpu().detach().numpy() / args.scales["sL"] if "sL" in args.infer_paras else case.sL_refe
        Ea = case.Ea_infe_s.cpu().detach().numpy() / args.scales["Ea"] if "Ea" in args.infer_paras else case.Ea

        self.YF_pred = YF_in + cp * (T_in - self.T_pred) / qF
        coef = sL + Rg * T_in / sL
        self.u_pred = 0.5 * (coef - (coef ** 2 - 4 * Rg * self.T_pred) ** 0.5)  # choose the smaller root (subsonic)
        self.rho_pred = rho_in * sL / self.u_pred
        self.omega_pred = A * np.exp(-Ea / (R * self.T_pred)) * (self.YF_pred * self.rho_pred) ** nu_rxn
        self.p_pred = self.rho_pred * Rg * self.T_pred
        
        self.T_refe = case.func_T(self.x[:, None]).ravel()
        self.YF_refe = case.func_YF(self.x[:, None]).ravel()
        self.u_refe = case.func_u(self.x[:, None]).ravel()
        self.rho_refe = case.func_rho(self.x[:, None]).ravel()
        self.omega_refe = case.func_omega(self.x[:, None]).ravel()
        self.p_refe = case.func_p(self.x[:, None]).ravel()
        
        self.preds += [self.T_pred, self.YF_pred, self.u_pred, self.rho_pred, self.omega_pred, self.p_pred - self.p_pred[0]]
        self.refes += [self.T_refe, self.YF_refe, self.u_refe, self.rho_refe, self.omega_refe, self.p_refe - self.p_refe[0]]
        self.mathnames += ["$T$", "$Y_F$", "$u$", r"$\rho$", r"$\omega$", "$p_{rel}$"]
        self.textnames += ["T", "YF", "u", "rho", "omega", "p"]
        self.units += ["K", " ", "m/s", "kg/m$^3$", "kg/(m$^3$·s)", "Pa"]

        if "sL" in args.infer_paras:
            self.para_infes += [case.sL_infe_s / args.scales["sL"], ]
            self.para_refes += [case.sL_refe, ]
            self.para_mathnames += ["$s_L$", ]
            self.para_textnames += ["sL", ]
            self.para_units += ["m/s", ]

        if "lam" in args.infer_paras:
            self.para_infes += [case.lam_infe_s / args.scales["lam"], ]
            self.para_refes += [case.lam, ]
            self.para_mathnames += [r"$\lambda$", ]
            self.para_textnames += ["lambda", ]
            self.para_units += ["W/(m·K)", ]

        if "Ea" in args.infer_paras:
            self.para_infes += [case.Ea_infe_s / args.scales["Ea"], ]
            self.para_refes += [case.Ea, ]
            self.para_mathnames += ["$E_a$", ]
            self.para_textnames += ["Ea", ]
            self.para_units += ["J/mol", ]
