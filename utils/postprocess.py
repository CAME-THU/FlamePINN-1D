import deepxde as dde
import numpy as np
import torch
import os
from utils import metric_funcs

import matplotlib
matplotlib.use("Agg")  # do not show figures
import matplotlib.pyplot as plt

set_fs = 22
set_dpi = 200
plt.rcParams["font.sans-serif"] = "Arial"  # default font
plt.rcParams["font.size"] = set_fs  # default font size
# plt.rcParams["mathtext.fontset"] = "stix"  # default font of math text


class PostProcess:
    """Base class for post-processing.

    Args:
        args: args of the main function.
        case: case file variable.
        model: deepxde model.
        output_dir: output directory.
    """

    def __init__(self, args, case, model, output_dir):
        self.args = args
        self.case = case
        self.model = model
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # The following 5 variables should be modified by inheritance, and they must have the same length.
        # Each item of preds and refes is ndarray, and their shape should match the problem dimension.
        # For example, (n_x, ) for 1D problems, (n_x, n_y) for 2D, (n_x, n_y, n_t) for 2Dt, etc.
        self.preds = []  # predicted field variables of interest
        self.refes = []  # corresponding reference field variables of interest
        self.mathnames = []  # the field variable names in math format, using for figure titles, for example
        self.textnames = []  # the field variable names in plain text format
        self.units = []  # the units of the field variables

    def _save_data(self, save_refe=True, suffix=""):
        """Save the predicted and reference fields."""
        print("Saving data...")
        output_dir = self.output_dir
        os.makedirs(output_dir + "data/", exist_ok=True)
        for i in range(len(self.preds)):
            np.save(output_dir + f"data/{self.textnames[i]}_{suffix}_predicted.npy", self.preds[i])
            if save_refe:
                np.save(output_dir + f"data/{self.textnames[i]}_{suffix}_reference.npy", self.refes[i])

    def _save_metrics(self, refes, preds, output_dir):
        """Save the evaluation metrics of given predicted results w.r.t. reference results."""
        refes_ = [field.ravel() for field in refes]
        preds_ = [field.ravel() for field in preds]

        formats = {"l2 relative error": "{:.4%}",
                   "l1 relative error": "{:.4%}",
                   "MSE": "{:.2e}",
                   "RMSE": "{:.2e}",
                   "MAE": "{:.2e}",
                   "MaxE": "{:.2e}",
                   "MAPE": "{:.4%}",
                   "R2": "{:.4f}",
                   "mean absolute of refe": "{:.2e}",
                   "mean absolute of pred": "{:.2e}",
                   "min absolute of refe": "{:.2e}",
                   "min absolute of pred": "{:.2e}",
                   "max absolute of refe": "{:.2e}",
                   "max absolute of pred": "{:.2e}",
                   }

        metrics = {}
        for key in formats.keys():
            metrics[key] = []
            # metric_func = dde.metrics.get(key)
            metric_func = metric_funcs.get(key)
            for i in range(len(self.preds)):
                metric = metric_func(refes_[i], preds_[i])
                metrics[key].append(formats[key].format(metric))

        file = open(output_dir + "metrics.txt", "a")
        # file = open(output_dir + "metrics.txt", "w")
        file.write("\n")
        file.write("field variables:   ")
        file.write(", ".join(self.textnames))
        file.write("\n")
        for key in metrics.keys():
            file.write(key + ":   ")
            file.write(", ".join(metrics[key]))
            file.write("\n")
        file.write("\n")
        file.close()

        return metrics

    def save_metrics(self):
        """Save the evaluation metrics of predicted results w.r.t. reference results."""
        print("Saving metrics...")
        self._save_metrics(self.refes, self.preds, self.output_dir)

    def save_var_metrics(self, vars_refe=(0.1,), vars_infe=(dde.Variable(1.0),), vars_name=("nu",)):
        """Save the evaluation metrics of external trainable variables."""
        print("Saving metrics of external trainable variables...")
        output_dir = self.output_dir
        file = open(output_dir + "metrics.txt", "a")
        # file = open(output_dir + "metrics.txt", "w")
        file.write("\n")
        file.write("external trainable variables:   ")
        file.write(", ".join(vars_name))
        file.write("\n")

        file.write("reference:   ")
        for i in range(len(vars_refe)):
            file.write("{:.4e}".format(vars_refe[i]) + ", ")
        file.write("\n")

        file.write("inferred:   ")
        for i in range(len(vars_infe)):
            file.write("{:.4e}".format(vars_infe[i]) + ", ")
        file.write("\n")

        file.write("relative error:   ")
        for i in range(len(vars_infe)):
            file.write("{:.4%}".format(vars_infe[i] / vars_refe[i] - 1) + ", ")
        file.write("\n")
        file.close()

    @staticmethod
    def _plot_save_loss_history(model, names, output_dir, save_name):
        """Plot the loss history and save the history data."""
        print("Plotting and saving loss history...")
        os.makedirs(output_dir + "pics/", exist_ok=True)

        loss_history = model.losshistory
        loss_train = np.array(loss_history.loss_train)
        loss_names = names["equations"] + names["ICBCOCs"]

        s_de = slice(0, len(names["equations"]), 1)
        s_bc = slice(len(names["equations"]), len(loss_names), 1)
        ss = [s_de, s_bc]

        # plot in two figures
        fig, axes = plt.subplots(1, 2, sharey="all", figsize=(15, 6))
        for i in range(2):
            axes[i].set_xlabel("Epoch")
            axes[i].set_yscale("log")
            axes[i].tick_params(axis="y", labelleft=True)
            axes[i].plot(loss_history.steps, loss_train[:, ss[i]])
            axes[i].legend(loss_names[ss[i]], fontsize="small")
        plt.savefig(output_dir + f"pics/{save_name}_2figs.png", bbox_inches="tight", dpi=set_dpi)
        plt.close(fig)

        # plot in one figure
        plt.figure(figsize=(8, 6))
        plt.xlabel("Epoch")
        plt.yscale("log")
        plt.plot(loss_history.steps, loss_train, lw=2)
        plt.legend(loss_names, fontsize="small")
        plt.savefig(output_dir + f"pics/{save_name}_1fig.png", bbox_inches="tight", dpi=set_dpi)
        plt.close(fig)

        # save the loss history
        loss_save = np.hstack([
            np.array(loss_history.steps)[:, None],
            np.array(loss_history.loss_train),
        ])
        np.savetxt(output_dir + f"{save_name}.txt", loss_save, fmt="%.2e", delimiter="    ",
                   header="    ".join(["step"] + loss_names))

    def plot_save_loss_history(self):
        self._plot_save_loss_history(self.model, self.case.names, self.output_dir, "losses_train")

    def plot_save_var_history(self, vars_refe=(0.1,), scales=(1,),
                              mathnames=(r"$\nu$",), textnames=("nu",), units=("m$^2$/s",)):
        """Plot the history of external trainable variables and save the history data."""
        print("Plotting and saving variable learning history...")
        output_dir = self.output_dir
        os.makedirs(output_dir + "data/", exist_ok=True)
        os.makedirs(output_dir + "pics/", exist_ok=True)

        file = open(self.output_dir + "vars_history_scaled.txt", "r")
        lines = file.readlines()
        epochs = np.array([int(line.split(" [")[0]) for line in lines])
        vars_history = np.array([line.split(" [")[1][:-2].split(", ") for line in lines], dtype=float)
        vars_history = vars_history / np.array(scales)
        file.close()

        for i in range(vars_history.shape[1]):
            plt.figure(figsize=(8, 6))
            plt.title(mathnames[i], fontsize="medium")
            plt.xlabel("Epoch")
            plt.ylabel(units[i])
            plt.plot(epochs, np.ones(len(epochs)) * vars_refe[i], c="k", ls="--", lw=3, label="Reference")
            plt.plot(epochs, vars_history[:, i], lw=2, label="PINN")
            plt.legend(fontsize="small")
            plt.savefig(output_dir + f"pics/variable{i + 1}_{textnames[i]}.png", bbox_inches="tight", dpi=set_dpi)
            plt.close()
        np.save(output_dir + "data/variables_history.npy", vars_history)

    def delete_old_models(self):
        """Delete the old models produced during training"""
        print("Deleting old models...")
        output_dir = self.output_dir
        model_list = os.listdir(output_dir + "models/")
        model_list_better = [s for s in model_list if "better" in s]
        better_epochs = [int(s.split("-")[1][:-3]) for s in model_list_better]
        best_epoch_index = better_epochs.index(max(better_epochs))
        for filename in model_list_better:
            if filename != model_list_better[best_epoch_index]:
                os.remove(output_dir + "models/" + filename)


class PostProcess1D(PostProcess):
    """Post-processing for 1D problems (1D space or 1D time)."""

    def __init__(self, args, case, model, output_dir):
        super().__init__(args, case, model, output_dir)
        self.n_x = 5001
        self.x_l, self.x_r = case.x_l, case.x_r
        self.x = np.linspace(self.x_l, self.x_r, self.n_x)  # can be modified to t_l if the only dimension is t
        self.x_name = "$x$"  # can be modified by inheritance, to $t$" for example
        self.x_unit = "m"  # can be modified by inheritance, to "s" for example

    def save_data(self, save_refe=True):
        """Save the predicted and reference fields."""
        self._save_data(save_refe=save_refe, suffix="1D")

    def plot_1dfields(self, lw=2, format="png", extra_plot=None):
        """Plot the curves of predicted 1D fields."""
        print("Plotting 1D fields...")
        output_dir = self.output_dir
        os.makedirs(output_dir + "pics/", exist_ok=True)
        x = self.x
        preds = self.preds
        mnames, tnames, units = self.mathnames, self.textnames, self.units

        for i in range(len(preds)):
            plt.figure(figsize=(8, 6))
            plt.title(mnames[i], fontsize="medium")
            plt.xlabel(f"{self.x_name}/{self.x_unit}")
            plt.ylabel(units[i])
            plt.plot(x, preds[i], lw=lw)
            if extra_plot is not None:
                extra_plot()
            plt.savefig(output_dir + f"pics/field{i + 1}_{tnames[i]}.{format}", bbox_inches="tight", dpi=set_dpi)
            plt.close()

    def plot_1dfields_comp(self, lws=(1, 2.5), label_refe="Reference", label_pred="PINN",
                           fsize_legend=set_fs - 6, format="png", extra_plot=None):
        """Plot the curves of predicted v.s. reference 1D fields."""
        print("Plotting 1D fields (predicted vs reference)...")
        output_dir = self.output_dir
        os.makedirs(output_dir + "pics/", exist_ok=True)
        x = self.x
        preds = self.preds
        refes = self.refes
        mnames, tnames, units = self.mathnames, self.textnames, self.units

        for i in range(len(preds)):
            plt.figure(figsize=(8, 6))
            plt.title(mnames[i], fontsize="medium")
            plt.xlabel(f"{self.x_name}/{self.x_unit}")
            plt.ylabel(units[i])
            plt.plot(x, refes[i], c="C0", ls="-", lw=lws[0], label=label_refe)
            plt.plot(x, preds[i], c="C1", ls="--", lw=lws[1], label=label_pred)
            plt.legend(fontsize=fsize_legend)
            if extra_plot is not None:
                extra_plot()
            plt.savefig(output_dir + f"pics/fieldComp{i + 1}_{tnames[i]}.{format}", bbox_inches="tight", dpi=set_dpi)
            plt.close()


class PostProcess1Dt(PostProcess):
    """Post-processing for 1D unsteady problems, i.e. the independent variables are x and t."""
    pass


class PostProcess2D(PostProcess):
    """Post-processing for 2D steady problems, i.e. the independent variables are x and y."""
    pass


class PostProcess2Dt(PostProcess):
    """Post-processing for 2D unsteady problems, i.e. the independent variables are x, y, and t."""
    pass
