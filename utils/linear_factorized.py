import torch
from torch import Tensor
from torch.nn.parameter import Parameter
# from .. import functional as F
# from .. import init
# from .module import Module
from torch.nn import functional as F
from torch.nn import init
from torch.nn.modules import Module
from typing import Callable, Union, Dict


class WeightFactorizedLinear(Module):
    """
    Linear layer with random weight factorization (RWF).
    Applies a linear transformation to the incoming data: :math:`y = xA^T + b`
    The weight A is factorized as :math: `A = diag(s)V`
    See https://doi.org/10.48550/arXiv.2210.01274 or https://doi.org/10.48550/arXiv.2308.08468.
    """
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    # weight: Tensor

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None,
                 kernel_initializer: Callable = init.xavier_normal_,
                 weight_fact: Union[None, Dict] = None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.kernel_initializer = kernel_initializer
        self.weight_fact = weight_fact

        if weight_fact is None:
            self.weight = Parameter(torch.empty((out_features, in_features), **factory_kwargs))
            kernel_initializer(self.weight)
            self.register_parameter('weight_s', None)
            self.register_parameter('weight_v', None)
        else:
            weight = torch.empty((out_features, in_features), **factory_kwargs)
            kernel_initializer(weight)
            mean, std = weight_fact["mean"], weight_fact["std"]
            weight_s = torch.normal(mean, std, size=(out_features,))
            weight_s = torch.exp(weight_s)
            weight_v = weight / weight_s[:, None]
            self.weight_s = Parameter(weight_s)
            self.weight_v = Parameter(weight_v)
            self.register_parameter('weight', None)

        if bias:
            self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
            init.zeros_(self.bias)
        else:
            self.register_parameter('bias', None)

    def forward(self, input: Tensor) -> Tensor:
        if self.weight_fact is None:
            weight = self.weight
        else:
            weight = self.weight_s[:, None] * self.weight_v
        return F.linear(input, weight, self.bias)

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}, weight_fact={}'.format(
            self.in_features, self.out_features, self.bias is not None, self.weight_fact is not None
        )

