import torch
from typing import Union, Callable, List, Tuple, Dict

# from .nn import NN
# from .. import activations
# from .. import initializers
# from ... import config
from deepxde.nn.pytorch.nn import NN
from deepxde.nn import activations
from deepxde.nn import initializers
from deepxde import config


class DebuggedFNN(NN):
    """
    Fully-connected neural network (i.e. multi-layer perception (MLP)).
    Original deepxde doesn't support input_transform for pytorch. 
    This bug is fixed here, by setting input_transform as an input parameter.
    """

    # def __init__(self, layer_sizes, activation, kernel_initializer):
    def __init__(self,
                 layer_sizes: Union[List, Tuple],
                 activation: Union[str, List] = "tanh",
                 kernel_initializer: Union[str, Callable] = "Glorot normal",
                 input_transform: Union[None, Callable] = None
                 ) -> None:
        super().__init__()

        if isinstance(activation, list):
            if not (len(layer_sizes) - 1) == len(activation):
                raise ValueError(
                    "Total number of activation functions do not match with sum of hidden layers and output layer!"
                )
            self.activation = list(map(activations.get, activation))
        else:
            self.activation = activations.get(activation)
        self.initializer = initializers.get(kernel_initializer)
        self.initializer_zero = initializers.get("zeros")

        self._input_transform = input_transform  # add
        dim_input = layer_sizes[0]  # add
        if input_transform is not None:  # add
            test_x = torch.randn([1, dim_input], dtype=config.real(torch))
            dim_input_transform = input_transform(test_x).shape[1]
        else:
            dim_input_transform = dim_input
        self.dim_input_transform = dim_input_transform

        self.linears = torch.nn.ModuleList()
        for i in range(1, len(layer_sizes)):  # modified
            # self.linears.append(
            #     torch.nn.Linear(
            #         layer_sizes[i - 1], layer_sizes[i], dtype=config.real(torch)
            #     )
            # )
            prev_layer_size = layer_sizes[i - 1] if i != 1 else dim_input_transform
            curr_layer_size = layer_sizes[i]
            self.linears.append(torch.nn.Linear(
                prev_layer_size, curr_layer_size, dtype=config.real(torch)
            ))
            self.initializer(self.linears[-1].weight)
            self.initializer_zero(self.linears[-1].bias)

    def forward(self, inputs):
        x = inputs
        if self._input_transform is not None:
            x = self._input_transform(x)

        for j, linear in enumerate(self.linears[:-1]):
            x = (
                self.activation[j](linear(x))
                if isinstance(self.activation, list)
                else self.activation(linear(x))
            )
        x = self.linears[-1](x)

        if self._output_transform is not None:
            x = self._output_transform(inputs, x)
        return x

