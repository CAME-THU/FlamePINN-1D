"""Network architecture, input transform, and output transform."""
# import deepxde as dde
import torch
# import numpy as np
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
        xs = (x + self.args.shifts["x"]) * self.args.scales["x"]
        inputs = torch.cat([
            xs,
            ], dim=1)
        return inputs

    def output_denorm_transform(self, x, Ts):
        T = Ts / self.args.scales["T"]
        return T

    def output_physical_transform(self, x, Ts):
        T_minlim = self.case.T_in
        T_maxlim = self.case.T_max
        T = torch.sigmoid(Ts) * (T_maxlim - T_minlim) + T_minlim
        return T

