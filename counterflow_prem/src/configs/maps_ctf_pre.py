"""Network architecture, input transform, and output transform."""
# import deepxde as dde
import torch
import torch.nn.functional as F
# import numpy as np
# import math as mt
from utils.networks import FNN, ModifiedMLP

# dtype = dde.config.real(torch)
dtype = torch.float64

R = 8314.46261815324  # J/kmol/K


class Maps:
    def __init__(self, args, case):
        self.args = args
        self.case = case

        n_spe = args.gas.n_species
        # self.net = FNN(
        self.net = ModifiedMLP(
            layer_sizes=[1] + 6 * [64] + [3 + n_spe],
            activation="tanh",  # "tanh", "sin"
            kernel_initializer="Glorot normal",
            weight_fact={"mean": 0.5, "std": 0.1},
            input_transform=self.input_transform,
        )

        if args.bc_type == "hard":
            self.net.apply_output_transform(self.output_hardbc_transform)
        else:
            # self.net.apply_output_transform(self.output_denorm_transform)
            self.net.apply_output_transform(self.output_physical_transform)

    def input_transform(self, x):
        xs = (x + self.args.shifts["x"]) * self.args.scales["x"]
        inputs = torch.cat([
            xs,
            ], dim=1)
        return inputs

    def output_denorm_transform(self, x, uVTYs_s):
        us, Vs, Ts, Yss = uVTYs_s[:, 0:1], uVTYs_s[:, 1:2], uVTYs_s[:, 2:3], uVTYs_s[:, 3:]
        u = us / self.args.scales["u"]
        V = Vs / self.args.scales["V"]
        T = Ts / self.args.scales["T"]
        scale_Ys = torch.tensor(self.args.scales["Ys"], dtype=dtype)
        Ys = Yss / scale_Ys
        return torch.cat([u, V, T, Ys], dim=1)

    def output_physical_transform(self, x, uVTYs_s):
        us, Vs, Ts, Yss = uVTYs_s[:, 0:1], uVTYs_s[:, 1:2], uVTYs_s[:, 2:3], uVTYs_s[:, 3:]
        u = us / self.args.scales["u"]
        V = Vs / self.args.scales["V"]

        T_minlim, T_maxlim = min(self.case.T_u, self.case.T_b), 3000
        T = torch.sigmoid(Ts) * (T_maxlim - T_minlim) + T_minlim

        Ys = F.softmax(Yss, dim=1)

        return torch.cat([u, V, T, Ys], dim=1)

    def output_hardbc_transform(self, x, uVTYs_s):
        pass
        # TODO
