"""Network architecture, input transform, and output transform."""
# import deepxde as dde
import torch
# import torch.nn.functional as F
# import numpy as np
# import math as mt
from utils.networks import DebuggedFNN as FNN

# dtype = dde.config.real(torch)
dtype = torch.float64


class Maps:
    def __init__(self, args, case):
        self.args = args
        self.case = case

        self.net = FNN(
            layer_sizes=[1] + 3 * [64] + [1],
            activation="tanh",  # "tanh", "sin"
            kernel_initializer="Glorot normal",
            input_transform=self.input_transform,
        )

        # self.net.apply_output_transform(self.output_denorm_transform)
        self.net.apply_output_transform(self.output_physical_transform)

    def input_transform(self, x):
        xs = (x + self.args.shift_x) * self.args.scale_x
        inputs = torch.cat([
            xs,
            ], dim=1)
        return inputs

    def output_denorm_transform(self, x, Ts):
        T = Ts / self.args.scale_T
        return T

    def output_physical_transform(self, x, Ts):
        T_minlim = self.case.T_in
        T_maxlim = self.case.T_max
        # T = torch.log(torch.tensor(1) + torch.exp(Ts)) + T_minlim
        T = torch.sigmoid(Ts) * (T_maxlim - T_minlim) + T_minlim
        # T = (torch.sin(Ts) + 1) / 2 * (T_maxlim - T_minlim) + T_minlim
        return T

