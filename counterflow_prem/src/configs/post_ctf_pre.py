"""Define the field variables to be post-processed."""
import numpy as np
import os
from utils.postprocess import PostProcess1D

import matplotlib
matplotlib.use("Agg")  # do not show figures
import matplotlib.pyplot as plt
set_fs = 22
set_dpi = 200
plt.rcParams["font.sans-serif"] = "Arial"  # default font
plt.rcParams["font.size"] = set_fs  # default font size
# plt.rcParams["mathtext.fontset"] = "stix"  # default font of math text


class PostProcessFlame(PostProcess1D):
    def __init__(self, args, case, model, output_dir):
        super().__init__(args, case, model, output_dir)

        # ----------------------------------------------------------------------
        # Get the predicted and reference fields
        output = model.predict(self.x[:, None])
        self.u_pred = output[:, 0]
        self.V_pred = output[:, 1]
        self.T_pred = output[:, 2]
        self.Ys_pred = output[:, 3:]
        R = 8314.46261815324  # J/kmol/K
        W_pred = 1 / np.matmul(self.Ys_pred, 1 / args.gas.molecular_weights)
        self.rho_pred = case.p * W_pred / (R * self.T_pred)

        self.u_refe = case.func_u(self.x[:, None]).ravel()
        self.V_refe = case.func_V(self.x[:, None]).ravel()
        self.T_refe = case.func_T(self.x[:, None]).ravel()
        self.Ys_refe = case.func_Ys(self.x[:, None])
        self.rho_refe = case.func_rho(self.x[:, None]).ravel()

        n_spe = args.gas.n_species
        spe_names = args.gas.species_names
        self.preds += [self.u_pred, self.V_pred, self.T_pred] + [self.Ys_pred[:, k] for k in range(n_spe)] + [self.rho_pred]
        self.refes += [self.u_refe, self.V_refe, self.T_refe] + [self.Ys_refe[:, k] for k in range(n_spe)] + [self.rho_refe]
        self.mathnames += ["$u$", "$V$", "$T$"] + [rf"$Y_{{\rm {spe_names[k]}}}$" for k in range(n_spe)] + [r"$\rho$"]
        self.textnames += ["u", "V", "T"] + ["Y_" + spe_names[k] for k in range(n_spe)] + ["rho"]
        self.units += ["m/s", "s$^{-1}$", "K"] + [""] * n_spe + ["kg/m$^3$"]
        
        if "pCurv" in args.infer_paras:
            self.para_infes += [case.pCurv_infe_s / args.scales["pCurv"], ]
            self.para_refes += [case.pCurv_refe, ]
            self.para_mathnames += [r"$\Lambda$", ]
            self.para_textnames += ["Lambda", ]
            self.para_units += ["Pa/m$^2$", ]
            
        if "Ea" in args.infer_paras:
            self.para_infes += [case.Ea_infe_s / args.scales["Ea"], ]
            self.para_refes += [case.Ea, ]
            self.para_mathnames += ["$E_a$", ]
            self.para_textnames += ["Ea", ]
            self.para_units += ["J/kmol", ]

    def plot_species(self):
        """Plot the predicted species in one figure."""
        print("Plotting species...")
        args = self.args
        output_dir = self.output_dir
        os.makedirs(output_dir + "pics/", exist_ok=True)
        x = self.x

        n_spe = args.gas.n_species
        spe_names = args.gas.species_names
        plt.figure(figsize=(8, 6))
        plt.title("Mass Fractions", fontsize="medium")
        plt.xlabel("$x$/m")
        for k in range(n_spe):
            plt.plot(x, self.Ys_pred[:, k], lw=2, label=spe_names[k])
        plt.legend(fontsize=set_fs - 6)
        plt.savefig(output_dir + f"pics/field_Ys.png", bbox_inches="tight", dpi=set_dpi)
        plt.close()

