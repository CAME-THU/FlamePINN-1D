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
        self.T_pred = output[:, 0]
        self.Ys_pred = output[:, 1:]
        R = 8314.46261815324  # J/kmol/K
        W_pred = 1 / np.matmul(self.Ys_pred, 1 / args.gas.molecular_weights)
        self.rho_pred = case.p * W_pred / (R * self.T_pred)
        sL = case.sL_infe_s.cpu().detach().numpy() / args.scales["sL"] if "sL" in args.infer_paras else case.sL_refe
        self.u_pred = self.rho_pred[0] * sL / self.rho_pred
        # self.u_pred = self.rho_pred[0] * case.sL_refe / self.rho_pred

        self.T_refe = case.func_T(self.x[:, None]).ravel()
        self.Ys_refe = case.func_Ys(self.x[:, None])
        self.u_refe = case.func_u(self.x[:, None]).ravel()
        self.rho_refe = case.func_rho(self.x[:, None]).ravel()

        n_spe = args.gas.n_species
        spe_names = args.gas.species_names
        self.preds += [self.T_pred] + [self.Ys_pred[:, k] for k in range(n_spe)] + [self.u_pred, self.rho_pred]
        self.refes += [self.T_refe] + [self.Ys_refe[:, k] for k in range(n_spe)] + [self.u_refe, self.rho_refe]
        self.mathnames += ["$T$"] + [rf"$Y_{{\rm {spe_names[k]}}}$" for k in range(n_spe)] + ["$u$", r"$\rho$"]
        self.textnames += ["T"] + ["Y_" + spe_names[k] for k in range(n_spe)] + ["u", "rho"]
        self.units += ["K"] + [""] * n_spe + ["m/s", "kg/m$^3$"]

        if "sL" in args.infer_paras:
            self.para_infes += [case.sL_infe_s / args.scales["sL"], ]
            self.para_refes += [case.sL_refe, ]
            self.para_mathnames += ["$s_L$", ]
            self.para_textnames += ["sL", ]
            self.para_units += ["m/s", ]

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

