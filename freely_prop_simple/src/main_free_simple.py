"""PINN to solve the forward problems of 1D freely-propagating premixed flames based on simplified physical models."""
import deepxde as dde
import numpy as np
# import torch
import os
import argparse

# # relative import is not suggested
# # set projects_deepxde as root (by clicking), or use sys.path.insert like this (then ignore the red errors):
# import sys
# sys.path.insert(0, os.path.dirname("E:/.../projects_PINN/"))
from configs.case_free_simple import Case
from configs.maps_T import Maps
from configs.post_free_simple import PostProcessFlame
from utils.utils import efmt, cal_stat
from utils.dataset_modi import ScaledDataSet


def main(args):
    case = Case(args)
    case_name = "p{:.2f}_phi{:.4f}".format(case.p_in/101325, case.phi)

    scale_T = args.scale_T
    scale_x, shift_x = args.scale_x, args.shift_x

    x_l, x_r = case.x_l, case.x_r

    # ----------------------------------------------------------------------
    # Z-shape warmup
    n_wms = [30, 20, 50]
    wm_x = np.hstack([
        np.linspace(x_l, x_l + 0.3 * (x_r - x_l), n_wms[0], endpoint=False),
        np.linspace(x_l + 0.3 * (x_r - x_l), x_l + 0.5 * (x_r - x_l), n_wms[1], endpoint=False),
        np.linspace(x_l + 0.5 * (x_r - x_l), x_r, n_wms[2])
    ])[:, None]
    wm_T = np.hstack([
        np.ones(n_wms[0]) * case.T_in,
        np.linspace(case.T_in, case.T_max, n_wms[1]),
        np.ones(n_wms[2]) * case.T_max
    ])[:, None]

    data_warm = ScaledDataSet(
        X_train=wm_x,
        y_train=wm_T,
        X_test=wm_x,
        y_test=wm_T,
        scales=(scale_T, ),
        # standardize=True,
    )

    # ----------------------------------------------------------------------
    # define sampling points
    n_dmn = 1001
    n_bdr = 4
    data = dde.data.PDE(
        case.geom,
        case.ode,
        case.icbcocs,
        num_domain=n_dmn,
        num_boundary=n_bdr,
        train_distribution="Hammersley",  # "Hammersley", "uniform", "pseudo"
        num_test=1000,
    )

    # ----------------------------------------------------------------------
    # define maps (network and input/output transforms)
    maps = Maps(args=args, case=case)
    net = maps.net
    model_warm = dde.Model(data_warm, net)
    model = dde.Model(data, net)

    # ----------------------------------------------------------------------
    # define training and train
    output_dir = f"../results/{case_name}/{args.problem_type}/"
    output_dir += f"dmn{n_dmn}"
    if args.bc_type == "soft":
        output_dir += f"_bdr{n_bdr}"
    if args.oc_type == "soft":
        output_dir += f"_ob{case.n_ob}-N{efmt(args.noise_level)}"

    output_dir += f"_T{efmt(scale_T)}_x{efmt(scale_x)}_x{efmt(shift_x)}"

    if not args.know_sL:
        output_dir += f"_sL{efmt(args.sL_ini)}"
    else:
        output_dir += f"_sL-known"

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

    if not args.know_sL:
        external_trainable_variables = [case.var_sL_s, ]
        variable_saver = dde.callbacks.VariableValue(
            external_trainable_variables, period=100, precision=4, filename=output_dir + "vars_history_scaled.txt")
        callbacks.append(variable_saver)
    else:
        external_trainable_variables = []

    # stage 1: warmup pretraining
    loss_weights = None
    n_iter_s1 = 1000
    model_warm.compile("adam", 1e-3, decay=("step", 1000, 0.95), loss_weights=loss_weights,
                       external_trainable_variables=external_trainable_variables)
    model_warm.train(
        iterations=n_iter_s1,
        display_every=100,
        model_restore_path=None,
        model_save_path=output_dir + "models/model_stage1_last", )

    os.makedirs(output_dir + "stage1/", exist_ok=True)
    pp1d_s1 = PostProcessFlame(args=args, case=case, model=model_warm, output_dir=output_dir + "stage1/")
    pp1d_s1.plot_save_loss_history()
    pp1d_s1.plot_1dfields_comp(lws=(2.5, 3.5))
    pp1d_s1.save_metrics()

    # stage 2: pure ODE training
    loss_weights = None
    # loss_weights = [1, 1, 100]
    model.compile("adam", 1e-3, decay=("step", 1000, 0.95), loss_weights=loss_weights,
                  external_trainable_variables=external_trainable_variables)
    print("[" + ", ".join(case.names["equations"] + case.names["ICBCOCs"]) + "]" + "\n")
    model.train(iterations=args.n_iter,
                display_every=100,
                disregard_previous_best=False,
                callbacks=callbacks,
                # model_restore_path=None,
                model_restore_path=output_dir + f"models/model_stage1_last-{n_iter_s1}.pt",
                model_save_path=output_dir + "models/model_last", )

    # ----------------------------------------------------------------------
    # restore a new model
    # model.restore(output_dir + "models/model_last-30000.pt")

    # ----------------------------------------------------------------------
    # post-process
    pp1d = PostProcessFlame(args=args, case=case, model=model, output_dir=output_dir)
    pp1d.save_data()
    pp1d.save_metrics()
    pp1d.plot_save_loss_history()
    if not args.know_sL:
        pp1d.save_var_metrics((case.sL_refe, ), (case.var_sL_s / args.scale_sL, ), ("sL", ))
        pp1d.plot_save_var_history((case.sL_refe, ), (args.scale_sL, ), ("$s_L$", ), ("sL", ), ("m/s", ))
    pp1d.delete_old_models()
    # pp1d.plot_1dfields()
    pp1d.plot_1dfields_comp(lws=(2.5, 3.5))

    return output_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hyper-parameters")
    parser.add_argument("--case_id", type=int, default=1)

    parser.add_argument("--problem_type", type=str, default="forward", help="options: forward, inverse")
    parser.add_argument("--bc_type", type=str, default="soft", help="options: none, soft")
    parser.add_argument("--oc_type", type=str, default="none", help="options: none, soft")

    parser.add_argument("--p_in", type=float, default=101325)
    parser.add_argument("--phi", type=float, default=0.42)

    parser.add_argument("--scale_T", type=float, default=1.0)

    parser.add_argument("--scale_x", type=float, default=1.0)
    parser.add_argument("--shift_x", type=float, default=0.0)

    parser.add_argument("--noise_level", type=float, default=0.00,
                        help="noise level for observations, such as 0.02 (2%)")

    parser.add_argument("--know_sL", type=bool, default=False)
    parser.add_argument("--sL_ini", type=float, default=0.5)
    parser.add_argument("--scale_sL", type=float, default=1)

    parser.add_argument("--n_iter", type=int, default=20000)
    parser.add_argument("--i_run", type=int, default=1)

    # ----------------------------------------------------------------------
    # set arguments
    args = parser.parse_args()

    args.case_id = 1
    args.p_in = 101325 * 1.2
    args.phi = 0.40

    width = 0.0015
    args.scale_T = 1 / 298.0  # may be changed in the case file
    # args.scale_x, args.shift_x = 1 / width, 0  # scale to [0, 1]
    args.scale_x, args.shift_x = 10 / width, 0  # scale to [0, 10]
    # args.scale_x, args.shift_x = 20 / width, -width / 2  # scale to [-10, 10]

    args.problem_type, args.bc_type, args.oc_type = "forward", "soft", "none"
    # args.problem_type, args.bc_type, args.oc_type = "inverse", "none", "soft"

    args.know_sL = False
    # args.know_sL = True
    # args.sL_ini = 0.1  # tends to flashback
    args.sL_ini = 0.4  # tends to blow out

    # args.n_iter = 1000
    # args.n_iter = 10000
    args.n_iter = 30000

    # ----------------------------------------------------------------------
    # run

    # n_run = 1
    # for args.i_run in range(1, 1 + n_run):
    #     output_dir = main(args)

    n_run = 1
    for args.p_in in (101325, 101325 * 1.2):
        for args.phi in (0.40, 0.42, 0.44, 0.46, 0.48, 0.50):
            for args.i_run in range(1, 1 + n_run):
                output_dir = main(args)


