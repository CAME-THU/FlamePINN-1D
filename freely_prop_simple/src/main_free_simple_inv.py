"""PINN to solve the inverse problems of 1D freely-propagating premixed flames based on simplified physical models."""
import deepxde as dde
import numpy as np
# import torch
import os
import time
import argparse

# # relative import is not suggested
# # set projects_deepxde as root (by clicking), or use sys.path.insert like this (then ignore the red errors):
# import sys
# sys.path.insert(0, os.path.dirname("E:/Research_ASUS/1 PhD project/AI_PDE/projects_PINN/"))
from configs.case_free_simple import Case
from configs.maps_free_simple import Maps
from configs.post_free_simple import PostProcessFlame
from utils.utils import efmt, cal_stat
# from utils.dataset_modi import ScaledDataSet
from utils.callbacks_modi import VariableSaver


def main(args):
    case = Case(args)
    # case_name = f"case{args.case_id}"
    # case_name = "p{:.2f}_phi{:.2f}".format(case.p_in/101325, case.phi)
    case_name = "p{:.2f}_T{:.0f}_phi{:.2f}".format(case.p_in/101325, case.T_in, case.phi)

    x_l, x_r = case.x_l, case.x_r

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
    # model_warm = dde.Model(data_warm, net)
    model = dde.Model(data, net)

    # ----------------------------------------------------------------------
    # define training and train
    output_dir = f"../results/{case_name}/{args.problem_type}/"
    output_dir += f"dmn{n_dmn}"
    if args.bc_type == "soft":
        output_dir += f"_bdr{n_bdr}"
    if args.oc_type == "soft":
        output_dir += f"_ob{case.n_ob}-N{efmt(args.noise_level)}"

    scale_T = args.scales["T"]
    scale_x, shift_x = args.scales["x"], args.shifts["x"]
    output_dir += f"_T{efmt(scale_T)}_x{efmt(scale_x)}_x{efmt(shift_x)}"

    if "sL" in args.infer_paras:
        # output_dir += f"_sL{efmt(args.scales['sL'])}"
        output_dir += f"_sL{efmt(args.infer_paras['sL'])}"
    else:
        output_dir += f"_sL-known"
    if "lam" in args.infer_paras:
        output_dir += f"_lam{efmt(args.infer_paras['lam'])}"
    if "Ea" in args.infer_paras:
        output_dir += f"_Ea{efmt(args.infer_paras['Ea'])}"

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

    external_trainable_variables, para_dict, var_saver = [], {}, None
    if "sL" in args.infer_paras:
        external_trainable_variables += [case.sL_infe_s, ]
        para_dict["sL"] = case.sL_infe_s
    if "lam" in args.infer_paras:
        external_trainable_variables += [case.lam_infe_s, ]
        para_dict["lam"] = case.lam_infe_s
    if "Ea" in args.infer_paras:
        external_trainable_variables += [case.Ea_infe_s, ]
        para_dict["Ea"] = case.Ea_infe_s
    if len(para_dict) > 0:
        var_saver = VariableSaver(para_dict, args.scales, period=100, filename=output_dir + "parameters_history.csv")
        callbacks.append(var_saver)

    loss_weights = None
    model.compile(optimizer="adam",  # "sgd", "rmsprop", "adam", "adamw"
                  lr=1e-3,
                  loss="MSE",
                  decay=("step", 1000, 0.95),
                  loss_weights=loss_weights,
                  external_trainable_variables=external_trainable_variables,
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
    pp1d.plot_save_loss_history()
    if len(args.infer_paras) > 0:
        pp1d.save_para_metrics()
        pp1d.plot_para_history(var_saver)
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
    parser.add_argument("--T_in", type=float, default=298.0)
    parser.add_argument("--phi", type=float, default=0.46)

    parser.add_argument("--scales", type=dict,
                        default={"x": 1.0, "T": 1.0, "sL": 1.0},
                        help="(variables * scale) for NN I/O, PDE scaling, and parameter inference")
    parser.add_argument("--shifts", type=dict, default={"x": 0.0},
                        help="((independent variables + shift) * scale) for NN input and PDE scaling")

    parser.add_argument("--infer_paras", type=dict, default={"sL": 0.4},
                        help="initial values for unknown physical parameters to be inferred")

    parser.add_argument("--n_ob", type=int, default=10, help="number of observation points for inverse problems")
    parser.add_argument("--noise_level", type=float, default=0.00,
                        help="noise level of observed data for inverse problems, such as 0.02 (2%)")
    parser.add_argument("--observe_u", type=bool, default=True)

    parser.add_argument("--n_iter", type=int, default=30000, help="number of training iterations")
    parser.add_argument("--i_run", type=int, default=1, help="index of the current run")

    # ----------------------------------------------------------------------
    # set arguments
    args = parser.parse_args()

    args.case_id = 1
    args.p_in = 101325 * 1.0
    args.T_in = 298.0
    args.phi = 0.46

    length = 0.0015
    args.scales["T"] = 1 / args.T_in  # may be changed in the case file
    args.scales["x"], args.shifts["x"] = 10 / length, 0  # scale to [0, 10]

    # args.infer_paras = {}  # known sL
    # args.infer_paras["sL"] = 0.1  # tends to flashback
    args.infer_paras["sL"] = 0.4  # tends to blow out

    # args.problem_type, args.bc_type, args.oc_type = "forward", "soft", "none"

    args.problem_type, args.bc_type, args.oc_type = "inverse", "none", "soft"
    args.infer_paras["lam"], args.scales["lam"] = 0.03, 10  # W/(m·K)
    args.infer_paras["Ea"], args.scales["Ea"] = 1.5e5, 1e-6  # J/mol

    args.n_ob = 10
    # args.noise_level = 0.0
    args.noise_level = 0.02

    args.observe_u = True

    # args.n_iter = 1000
    args.n_iter = 30000

    # ----------------------------------------------------------------------
    # run
    
    n_run = 1
    for args.i_run in range(1, 1 + n_run):
        print(args)
        output_dir = main(args)

    # n_run = 1
    # for args.infer_paras["sL"] in (0.4, ):
    #     for args.i_run in range(1, 1 + n_run):
    #         output_dir = main(args)

    # n_run = 1
    # for args.n_ob in (5, 6, 7, 8, 9, 10, 12, 15, 20, 25):
    #     for args.noise_level in (0.00, 0.01, 0.02):
    #         for args.i_run in range(1, 1 + n_run):
    #             output_dir = main(args)
    