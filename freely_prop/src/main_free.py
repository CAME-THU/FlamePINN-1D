"""PINN to solve the 1D freely-propagating premixed flames based on detailed physical models."""
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
from configs.case_free import Case
from configs.maps_free import Maps
from configs.post_free import PostProcessFlame
from utils.utils import efmt, cal_stat
from utils.gas1d import Gas1D, Gas1D_1stepIr
from utils.dataset_modi import ScaledDataSet
from utils.icbcs import ScaledPointSetBC
from utils.callbacks_modi import VariableSaver

dtype = torch.float64
# torch.set_default_dtype(dtype)
dde.config.real.set_float64()


def main(args):
    case = Case(args)
    mech_name = args.gas.source.split(".")[0]
    # case_name = mech_name
    case_name = mech_name + "/p{:.2f}_phi{:.2f}".format(case.p/101325, case.phi)

    scale_T, scale_Ys = args.scales["T"], args.scales["Ys"]
    scale_x, shift_x = args.scales["x"], args.shifts["x"]

    x_l, x_r = case.x_l, case.x_r

    n_spe = args.gas.n_species

    # ----------------------------------------------------------------------
    # define initial fields for warmup training

    # designed warmup
    n_wm = 501
    wm_x = np.linspace(x_l, x_r, n_wm)[:, None]
    wm_T = case.func_ini_guess(wm_x.ravel())[:, 0:1]
    wm_Ys = case.func_ini_guess(wm_x.ravel())[:, 1:]
    wm_TYs = np.hstack([wm_T, wm_Ys])
    data_warm = ScaledDataSet(
        X_train=wm_x,
        y_train=wm_TYs,
        X_test=wm_x,
        y_test=wm_TYs,
        scales=[scale_T] + [scale_Ys[k] for k in range(n_spe)],
        # standardize=True,
    )

    oc_T_wm = ScaledPointSetBC(wm_x, wm_T, component=0, scale=scale_T)
    ocs_Ys_wm = [ScaledPointSetBC(wm_x, wm_Ys[:, k:k+1], component=k+1, scale=scale_Ys[k]) for k in range(n_spe)]

    # ----------------------------------------------------------------------
    # define sampling points
    n_dmn = 100
    n_bdr = 4 if args.bc_type == "soft" else 0
    data_warm2 = dde.data.PDE(
        case.geom,
        case.ode,
        [oc_T_wm] + ocs_Ys_wm,
        num_domain=n_dmn,
        num_boundary=n_bdr,
        # train_distribution="uniform",  # "Hammersley", "uniform", "pseudo"
        train_distribution="pseudo",  # "Hammersley", "uniform", "pseudo"
        # anchors=ob_xy_s,
        # solution=func_sol,
        num_test=100,
    )
    data = dde.data.PDE(
        case.geom,
        case.ode,
        case.icbcocs,
        num_domain=n_dmn,
        num_boundary=n_bdr,
        # train_distribution="uniform",  # "Hammersley", "uniform", "pseudo"
        train_distribution="pseudo",  # "Hammersley", "uniform", "pseudo"
        num_test=100,
    )

    # ----------------------------------------------------------------------
    # define maps (network and input/output transforms)
    maps = Maps(args=args, case=case)
    net = maps.net
    model_warm = dde.Model(data_warm, net)
    model_warm2 = dde.Model(data_warm2, net)
    model = dde.Model(data, net)

    # ----------------------------------------------------------------------
    # define training and train
    output_dir = f"../results/{case_name}/{args.problem_type}/"
    output_dir += f"dmn{n_dmn}"
    if args.bc_type == "soft":
        output_dir += f"_bdr{n_bdr}"
    elif args.bc_type == "hard":
        output_dir += f"_hardBC"
    if args.oc_type == "soft":
        output_dir += f"_ob{case.n_ob}-N{efmt(args.noise_level)}"

    output_dir += f"_T{efmt(scale_T)}_x{efmt(scale_x)}_x{efmt(shift_x)}"

    if "sL" in args.infer_paras:
        output_dir += f"_sL{efmt(args.infer_paras['sL'])}"
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
    resampler = dde.callbacks.PDEPointResampler(period=10)
    callbacks += [resampler, ]

    external_trainable_variables, para_dict, var_saver = [], {}, None
    if "sL" in args.infer_paras:
        external_trainable_variables += [case.sL_infe_s, ]
        para_dict["sL"] = case.sL_infe_s
    if len(para_dict) > 0:
        var_saver = VariableSaver(para_dict, args.scales, period=100, filename=output_dir + "parameters_history.csv")
        callbacks.append(var_saver)

    # stage 1: warmup pretraining
    loss_weights = None
    n_iter_s1 = 5000
    model_warm.compile("adam", 1e-3, decay=("step", 1000, 0.95), loss_weights=loss_weights,
                       external_trainable_variables=external_trainable_variables)
    t0 = time.perf_counter()
    model_warm.train(
        iterations=n_iter_s1,
        display_every=100,
        model_restore_path=None,
        model_save_path=output_dir + "models/model_stage1_last", )
    t_took = time.perf_counter() - t0
    np.savetxt(output_dir + f"training_time_of_stage1_is_{t_took:.2f}s.txt", np.array([t_took]), fmt="%.2f")

    os.makedirs(output_dir + "stage1/", exist_ok=True)
    pp1d_s1 = PostProcessFlame(args=args, case=case, model=model_warm, output_dir=output_dir + "stage1/")
    # pp1d_s1.save_data()
    pp1d_s1.plot_save_loss_history()  # Note: the legend may be wrong
    # pp1d_s1.plot_1dfields()
    pp1d_s1.plot_1dfields_comp(lws=(2.5, 3.5), label_refe="Cantera")
    pp1d_s1.plot_species()
    pp1d_s1.save_metrics()
    data_warm, model_warm = None, None  # release memory

    # stage 2: warmup + ODE training
    loss_weights = None
    # loss_weights = [1] * len(case.names["equations"]) + [100] * len(data_warm2.bcs)
    n_iter_s2 = 1000
    model_warm2.compile("adam", 1e-3, decay=("step", 1000, 0.95), loss_weights=loss_weights,
                        external_trainable_variables=external_trainable_variables)
    t0 = time.perf_counter()
    model_warm2.train(
        iterations=n_iter_s2,
        display_every=100,
        callbacks=callbacks,
        # model_restore_path=None,
        model_restore_path=output_dir + f"models/model_stage1_last-{n_iter_s1}.pt",
        model_save_path=output_dir + "models/model_stage2_last", )
    t_took = time.perf_counter() - t0
    np.savetxt(output_dir + f"training_time_of_stage2_is_{t_took:.2f}s.txt", np.array([t_took]), fmt="%.2f")

    os.makedirs(output_dir + "stage2/", exist_ok=True)
    pp1d_s2 = PostProcessFlame(args=args, case=case, model=model_warm2, output_dir=output_dir + "stage2/")
    pp1d_s2.plot_save_loss_history()  # Note: the legend may be wrong
    # pp1d_s2.plot_1dfields()
    pp1d_s2.plot_1dfields_comp(lws=(2.5, 3.5), label_refe="Cantera")
    pp1d_s2.plot_species()
    pp1d_s2.save_metrics()
    data_warm2, model_warm2 = None, None  # release memory

    # stage 3: pure ODE training
    loss_weights = None
    # loss_weights = [1] * len(case.names["equations"]) + [100] * len(data.bcs)
    model.compile("adam", 1e-3, decay=("step", 1000, 0.95), loss_weights=loss_weights,
                  external_trainable_variables=external_trainable_variables,)
    print("[" + ", ".join(case.names["equations"] + case.names["ICBCOCs"]) + "]" + "\n")
    t0 = time.perf_counter()
    model.train(iterations=args.n_iter,
                display_every=100,
                disregard_previous_best=False,
                callbacks=callbacks,
                # model_restore_path=None,
                # model_restore_path=output_dir + f"models/model_stage1_last-{n_iter_s1}.pt",
                model_restore_path=output_dir + f"models/model_stage2_last-{n_iter_s2}.pt",
                model_save_path=output_dir + "models/model_last", )
    t_took = time.perf_counter() - t0
    np.savetxt(output_dir + f"training_time_of_stage3_is_{t_took:.2f}s.txt", np.array([t_took]), fmt="%.2f")

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
    pp1d.plot_1dfields_comp(lws=(2.5, 3.5), label_refe="Cantera")
    pp1d.plot_species()

    return output_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hyper-parameters")
    parser.add_argument("--case_id", type=int, default=1)

    parser.add_argument("--problem_type", type=str, default="forward", help="options: forward, inverse")
    parser.add_argument("--bc_type", type=str, default="soft", help="options: none, soft, hard")
    parser.add_argument("--oc_type", type=str, default="none", help="options: none, soft")

    parser.add_argument("--gas", default=ct.Solution("gri30.yaml"), help="mechanism")
    parser.add_argument("--gas1d", default=Gas1D(ct.Solution("gri30.yaml")), help="Gas class")
    parser.add_argument("--length", type=float, default=0.001, help="domain length [m]")
    parser.add_argument("--p", type=float, default=ct.one_atm, help="pressure [Pa]")
    parser.add_argument("--T_in", type=float, default=300.0, help="inlet temperature [K]")
    parser.add_argument("--phi", type=float, default=1.0, help="inlet equivalence ratio")
    parser.add_argument("--fuel", type=str, default="CH4:1", help="fuel composition")
    parser.add_argument("--oxidizer", type=str, default="O2:1, N2:3.76", help="oxidizer composition")
    
    parser.add_argument("--scales", type=dict,
                        default={"x": 1.0, "T": 1.0, "Ys": [1.0, ], "sL": 1.0},
                        help="(variables * scale) for NN I/O, PDE scaling, and parameter inference")
    parser.add_argument("--shifts", type=dict, default={"x": 0.0},
                        help="((independent variables + shift) * scale) for NN input and PDE scaling")

    parser.add_argument("--infer_paras", type=dict, default={"sL": 0.5},
                        help="initial values for unknown physical parameters to be inferred")

    parser.add_argument("--n_ob", type=int, default=10, help="number of observation points for inverse problems")
    parser.add_argument("--noise_level", type=float, default=0.00,
                        help="noise level of observed data for inverse problems, such as 0.02 (2%)")

    parser.add_argument("--n_iter", type=int, default=30000, help="number of training iterations")
    parser.add_argument("--i_run", type=int, default=1, help="index of the current run")

    # ----------------------------------------------------------------------
    # set arguments
    args = parser.parse_args()

    args.case_id = 1
    if args.case_id == 1:
        args.gas = ct.Solution("1S_CH4_MP1.yaml")
        args.gas.transport_model = "Mix"
        args.gas1d = Gas1D_1stepIr(args.gas)
        args.fuel, args.oxidizer = "CH4:1.0", "O2:1.0, N2:3.76"
        args.scales["Ys"] = [10, 10, 50, 10, 5]  # O2, H2O, CH4, CO2, N2 --may be changed in case file.
    else:
        pass
        # TODO: more cases.

    args.T_in = 300.0
    args.p, args.length, args.infer_paras["sL"] = 101325 * 1, 0.0025, 0.6  # for 1atm
    # args.p, args.length, args.infer_paras["sL"] = 101325 * 5, 0.001, 0.6  # for 5atm
    args.phi = 0.8

    args.scales["T"] = 1 / args.T_in  # may be changed in case file.
    args.scales["x"], args.shifts["x"] = 4 / args.length, -args.length / 2  # scale to [-2, 2]

    # args.infer_paras = {}  # known sL

    args.problem_type, args.bc_type, args.oc_type = "forward", "soft", "none"
    # args.problem_type, args.bc_type, args.oc_type = "inverse", "none", "soft"

    # args.n_iter = 300
    args.n_iter = 30000

    # ----------------------------------------------------------------------
    # run

    n_run = 1
    for args.i_run in range(1, 1 + n_run):
        output_dir = main(args)

    # n_run = 1
    # for args.phi in (0.6, 0.8, 1.0, 1.2, 1.4, 1.6):
    #     for args.i_run in range(1, 1 + n_run):
    #         output_dir = main(args)

