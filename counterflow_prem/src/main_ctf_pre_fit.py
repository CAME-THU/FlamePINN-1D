"""Data fitting by NN. The ODE losses are absent."""
import deepxde as dde
import numpy as np
import cantera as ct
import torch
import os
import time
import argparse

# # relative import is not suggested
# # set projects_deepxde as root (by clicking), or use sys.path.insert like this (then ignore the red errors):
# import sys
# sys.path.insert(0, os.path.dirname("E:/Research_ASUS/1 PhD project/AI_PDE/projects_PINN/"))
from configs.case_ctf_pre import Case
from configs.maps_ctf_pre import Maps
from configs.post_ctf_pre import PostProcessFlame
from utils.utils import efmt, cal_stat
from utils.gas1d import Gas1D, Gas1D_1stepIr
from utils.dataset_modi import ScaledDataSet

# dtype = torch.float64
# torch.set_default_dtype(dtype)
dde.config.real.set_float64()


def main(args):
    case = Case(args)
    mech_name = args.gas.source.split(".")[0]
    # case_name = mech_name
    case_name = mech_name + "/p{:.2f}_phi{:.2f}".format(case.p/101325, case.phi)

    scale_u, scale_V, scale_T, scale_Ys = args.scales["u"], args.scales["V"], args.scales["T"], args.scales["Ys"]
    scale_x, shift_x = args.scales["x"], args.shifts["x"]

    x_l, x_r = case.x_l, case.x_r

    n_spe = args.gas.n_species

    # ----------------------------------------------------------------------
    # define observation points
    n_ob = args.n_ob
    ob_x = np.linspace(x_l, x_r, n_ob)[:, None]

    ob_u = case.func_u(ob_x)
    ob_V = case.func_V(ob_x)
    ob_T = case.func_T(ob_x)
    ob_Ys = case.func_Ys(ob_x)

    normal_noise_u = np.random.randn(len(ob_u))[:, None]
    normal_noise_V = np.random.randn(len(ob_V))[:, None]
    normal_noise_T = np.random.randn(len(ob_T))[:, None]
    normal_noise_Ys = np.random.randn(ob_Ys.size).reshape([ob_Ys.shape[0], ob_Ys.shape[1]])
    ob_u += normal_noise_u * ob_u * args.noise_level
    ob_V += normal_noise_V * ob_V * args.noise_level
    ob_T += normal_noise_T * ob_T * args.noise_level
    ob_Ys += normal_noise_Ys * ob_Ys * args.noise_level

    ob_uVTYs = np.hstack([ob_u, ob_V, ob_T, ob_Ys])
    data = ScaledDataSet(
        X_train=ob_x,
        y_train=ob_uVTYs,
        X_test=ob_x,
        y_test=ob_uVTYs,
        # standardize=True,
        scales=[scale_u, scale_V, scale_T] + [scale_Ys[k] for k in range(n_spe)],
    )

    # ----------------------------------------------------------------------
    # define maps (network and input/output transforms)
    maps = Maps(args=args, case=case)
    net = maps.net
    model = dde.Model(data, net)

    # ----------------------------------------------------------------------
    # define training and train
    output_dir = f"../results/{case_name}/fit/"
    output_dir += f"ob{n_ob}-N{efmt(args.noise_level)}"

    # output_dir += f"_u{efmt(scale_u)}-V{efmt(scale_V)}-T{efmt(scale_T)}_x{efmt(scale_x)}_x{efmt(shift_x)}"
    output_dir += f"_x{efmt(scale_x)}_x{efmt(shift_x)}"

    i_run = args.i_run
    while True:
        if not os.path.exists(output_dir + f"/{i_run}/"):
            output_dir += f"/{i_run}/"
            os.makedirs(output_dir)
            os.makedirs(output_dir + "models/")
            break
        else:
            i_run += 1

    model_saver = dde.callbacks.ModelCheckpoint(
        output_dir + "models/model_better", save_better_only=True, period=100)
    callbacks = [model_saver, ]

    loss_weights = None
    model.compile(optimizer="adam",  # "sgd", "rmsprop", "adam", "adamw"
                  lr=1e-3,
                  loss="MSE",
                  decay=("step", 1000, 0.95),
                  loss_weights=loss_weights,
                  )

    t0 = time.perf_counter()
    model.train(iterations=args.n_iter,
                display_every=100,
                disregard_previous_best=False,
                callbacks=callbacks,
                model_restore_path=None,
                model_save_path=output_dir + "models/model_last",)
    t_took = time.perf_counter() - t0
    np.savetxt(output_dir + f"training_time_is_{t_took:.2f}s.txt", np.array([t_took]), fmt="%.2f")

    # ----------------------------------------------------------------------
    # restore the best model (do not if using LBFGS)
    model_list = os.listdir(output_dir + "models/")
    model_list_better = [s for s in model_list if "better" in s]
    saved_epochs = [int(s.split("-")[1][:-3]) for s in model_list_better]
    best_epoch = max(saved_epochs)
    model.restore(output_dir + f"models/model_better-{best_epoch}.pt")

    # ----------------------------------------------------------------------
    # post-process
    pp1d = PostProcessFlame(args=args, case=case, model=model, output_dir=output_dir)
    pp1d.save_data()
    pp1d.save_metrics()
    pp1d.plot_save_loss_history()  # Note: the legend may be wrong
    # if len(args.infer_paras) > 0:
    #     pp1d.save_para_metrics()
    #     pp1d.plot_para_history(var_saver)
    pp1d.delete_old_models()
    # pp1d.plot_1dfields()
    pp1d.plot_1dfields_comp(lws=(2.5, 3.5), label_refe="Cantera")
    pp1d.plot_species()

    return output_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hyper-parameters")
    parser.add_argument("--case_id", type=int, default=1)

    # parser.add_argument("--problem_type", type=str, default="forward", help="options: forward, inverse")
    parser.add_argument("--problem_type", type=str, default="fit")
    parser.add_argument("--bc_type", type=str, default="none", help="options: none, soft, hard")
    parser.add_argument("--oc_type", type=str, default="soft", help="options: none, soft")

    parser.add_argument("--gas", default=ct.Solution("gri30.yaml"), help="mechanism")
    parser.add_argument("--gas1d", default=Gas1D(ct.Solution("gri30.yaml")), help="Gas class")
    parser.add_argument("--length", type=float, default=0.02, help="domain length [m]")
    parser.add_argument("--p", type=float, default=ct.one_atm, help="pressure [Pa]")
    parser.add_argument("--T_u", type=float, default=300.0, help="unburned-side inlet temperature [K]")
    parser.add_argument("--T_b", type=float, default=300.0, help="burned-side inlet temperature [K]")
    parser.add_argument("--mdot_u", type=float, default=1.0,
                        help="unburned-side inlet mass flow rate per unit area [kg/m^2/s]")
    parser.add_argument("--mdot_b", type=float, default=1.0,
                        help="burned-side inlet mass flow rate per unit area [kg/m^2/s]")
    # parser.add_argument("--Ys_u", type=list, default=[1.0, ], help="unburned-side inlet mass fractions")
    parser.add_argument("--phi", type=float, default=1.0, help="unburned-side inlet equivalence ratio")
    parser.add_argument("--fuel", type=str, default="CH4:1", help="fuel composition")
    parser.add_argument("--oxidizer", type=str, default="O2:1, N2:3.76", help="oxidizer composition")

    parser.add_argument("--scales", type=dict,
                        default={"x": 1.0, "u": 1.0, "V": 1.0, "T": 1.0, "Ys": [1.0, ], "pCurv": 1e-4},
                        help="(variables * scale) for NN I/O, PDE scaling, and parameter inference")
    parser.add_argument("--shifts", type=dict, default={"x": 0.0},
                        help="((independent variables + shift) * scale) for NN input and PDE scaling")

    parser.add_argument("--infer_paras", type=dict, default={"pCurv": -5000},
                        help="initial values for unknown physical parameters to be inferred")

    parser.add_argument("--n_ob", type=int, default=10, help="number of observation points for inverse problems")
    parser.add_argument("--noise_level", type=float, default=0.00,
                        help="noise level of observed data for inverse problems, such as 0.02 (2%)")

    parser.add_argument("--n_iter", type=int, default=20000, help="number of training iterations")
    parser.add_argument("--i_run", type=int, default=1, help="index of the current run")

    # ----------------------------------------------------------------------
    # set arguments
    args = parser.parse_args()

    args.case_id = 1
    if args.case_id == 1:
        args.gas = ct.Solution("1S_CH4_MP1.yaml")
        args.gas.transport_model = "Mix"
        args.gas1d = Gas1D_1stepIr(args.gas)
        args.mdot_u = 1.0  # kg/m^2/s
        args.mdot_b = 1.0  # kg/m^2/s
        args.fuel, args.oxidizer = "CH4:1.0", "O2:1.0, N2:3.76"
        args.scales["Ys"] = [10, 10, 50, 10, 1.]  # O2, H2O, CH4, CO2, N2 --may be changed in case file.
    else:
        pass
        # TODO: more cases.

    args.length = 0.02
    args.p = ct.one_atm
    args.T_u = 300.0
    args.T_b = 300.0
    args.phi = 0.8

    args.scales["u"], args.scales["V"], args.scales["T"] = 1., 0.01, 0.001  # may be changed in case file.
    args.scales["x"], args.shifts["x"] = 4 / args.length, -args.length / 2  # scale to [-2, 2]

    args.n_ob = 20
    args.noise_level = 0.00

    # args.n_iter = 200
    # args.n_iter = 1000
    args.n_iter = 50000

    # ----------------------------------------------------------------------
    # run
    n_run = 1
    for args.i_run in range(1, 1 + n_run):
        output_dir = main(args)
