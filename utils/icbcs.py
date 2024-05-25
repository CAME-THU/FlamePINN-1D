"""   """

import numbers
# import numpy as np

# from .. import backend as bkd
# from .. import config
# from .. import data
# from .. import gradients as grad
# from .. import utils
# from ..backend import backend_name

# import deepxde as dde
from deepxde import backend as bkd
# from deepxde import config
from deepxde import data
from deepxde import gradients as grad
# from deepxde import utils
# from deepxde.backend import backend_name

from deepxde.icbc import IC, DirichletBC, NeumannBC, RobinBC, PeriodicBC, PointSetBC, PointSetOperatorBC
from deepxde.icbc.boundary_conditions import backend_name


class ScaledIC(IC):
    """Initial conditions: y([x, t0]) = func([x, t0])."""

    def __init__(self, geom, func, on_initial, component=0, scale=1.0):
        super().__init__(geom, func, on_initial, component)
        self.scale = scale

    def error(self, X, inputs, outputs, beg, end, aux_var=None):
        values = self.func(X, beg, end, aux_var)
        if bkd.ndim(values) == 2 and bkd.shape(values)[1] != 1:
            raise RuntimeError(
                "IC function should return an array of shape N by 1 for each component."
                "Use argument 'component' for different output components."
            )
        return (outputs[beg:end, self.component: self.component + 1] - values) * self.scale


class ScaledDirichletBC(DirichletBC):
    """Dirichlet boundary conditions: y(x) = func(x)."""

    def __init__(self, geom, func, on_boundary, component=0, scale=1.0):
        super().__init__(geom, func, on_boundary, component)
        self.scale = scale

    def error(self, X, inputs, outputs, beg, end, aux_var=None):
        values = self.func(X, beg, end, aux_var)
        if bkd.ndim(values) == 2 and bkd.shape(values)[1] != 1:
            raise RuntimeError(
                "DirichletBC function should return an array of shape N by 1 for each "
                "component. Use argument 'component' for different output components."
            )
        return (outputs[beg:end, self.component: self.component + 1] - values) * self.scale


class ScaledNeumannBC(NeumannBC):
    """Neumann boundary conditions: dy/dn(x) = func(x)."""

    def __init__(self, geom, func, on_boundary, component=0, scale=1.0):
        super().__init__(geom, func, on_boundary, component)
        self.scale = scale

    def error(self, X, inputs, outputs, beg, end, aux_var=None):
        values = self.func(X, beg, end, aux_var)
        return (self.normal_derivative(X, inputs, outputs, beg, end) - values) * self.scale


class ScaledRobinBC(RobinBC):
    """Robin boundary conditions: dy/dn(x) = func(x, y)."""

    def __init__(self, geom, func, on_boundary, component=0, scale=1.0):
        super().__init__(geom, func, on_boundary, component)
        self.scale = scale

    def error(self, X, inputs, outputs, beg, end, aux_var=None):
        return (self.normal_derivative(X, inputs, outputs, beg, end) -
                self.func(X[beg:end], outputs[beg:end])
                ) * self.scale


class ScaledPeriodicBC(PeriodicBC):
    """Periodic boundary conditions on component_x."""

    def __init__(self, geom, component_x, on_boundary, derivative_order=0, component=0, scale=1.0):
        super().__init__(geom, component_x, on_boundary, derivative_order, component)
        self.scale = scale

    def error(self, X, inputs, outputs, beg, end, aux_var=None):
        mid = beg + (end - beg) // 2
        if self.derivative_order == 0:
            yleft = outputs[beg:mid, self.component: self.component + 1]
            yright = outputs[mid:end, self.component: self.component + 1]
        else:
            dydx = grad.jacobian(outputs, inputs, i=self.component, j=self.component_x)
            yleft = dydx[beg:mid]
            yright = dydx[mid:end]
        return (yleft - yright) * self.scale


class ScaledPointSetBC(PointSetBC):
    """Dirichlet boundary condition for a set of points.

    Compare the output (that associates with `points`) with `values` (target data).
    If more than one component is provided via a list, the resulting loss will
    be the addative loss of the provided componets.

    Args:
        points: An array of points where the corresponding target values are known and
            used for training.
        values: A scalar or a 2D-array of values that gives the exact solution of the problem.
        component: Integer or a list of integers. The output components satisfying this BC.
            List of integers only supported for the backend PyTorch.
        batch_size: The number of points per minibatch, or `None` to return all points.
            This is only supported for the backend PyTorch and PaddlePaddle.
            Note, If you want to use batch size here, you should also set callback
            'dde.callbacks.PDEPointResampler(bc_points=True)' in training.
        shuffle: Randomize the order on each pass through the data when batching.
    """

    def __init__(self, points, values, component=0, batch_size=None, shuffle=True, scale=1.0):
        super().__init__(points, values, component, batch_size, shuffle)
        self.scale = scale

    def error(self, X, inputs, outputs, beg, end, aux_var=None):
        slice_batch = slice(None) if self.batch_size is None else self.batch_indices
        slice_component = slice(self.component, self.component + 1
                                ) if isinstance(self.component, numbers.Number) else self.component
        term_outputs = outputs[beg:end, slice_component] * self.scale
        term_values = self.values[slice_batch] * self.scale
        return term_outputs - term_values


class ScaledPointSetOperatorBC(PointSetOperatorBC):
    """General operator boundary conditions for a set of points.

    Compare the function output, func, (that associates with `points`)
        with `values` (target data).

    Args:
        points: An array of points where the corresponding target values are
            known and used for training.
        values: An array of values which output of function should fulfill.
        func: A function takes arguments (`inputs`, `outputs`, `X`)
            and outputs a tensor of size `N x 1`, where `N` is the length of
            `inputs`. `inputs` and `outputs` are the network input and output
            tensors, respectively; `X` are the NumPy array of the `inputs`.
    """

    def __init__(self, points, values, func, batch_size=None, shuffle=True, scale=1.0):
        super().__init__(points, values, func)
        self.batch_size = batch_size
        self.scale = scale

        if batch_size is not None:  # batch iterator and state
            if backend_name not in ["pytorch", "paddle"]:
                raise RuntimeError(
                    "batch_size only implemented for pytorch and paddle backend"
                )
            self.batch_sampler = data.sampler.BatchSampler(len(self), shuffle=shuffle)
            self.batch_indices = None

    def __len__(self):
        return self.points.shape[0]

    def collocation_points(self, X):
        if self.batch_size is not None:
            self.batch_indices = self.batch_sampler.get_next(self.batch_size)
            return self.points[self.batch_indices]
        return self.points

    def error(self, X, inputs, outputs, beg, end, aux_var=None):
        slice_batch = slice(None) if self.batch_size is None else self.batch_indices
        # term_func = self.func(inputs[beg:end], outputs[beg:end], X[beg:end]) * self.scale
        term_func = self.func(inputs, outputs, X)[beg:end] * self.scale
        term_values = self.values[slice_batch] * self.scale
        return term_func - term_values


class ModifiedPointSetBC(PointSetBC):
    """Dirichlet boundary condition for a set of points.

    Compare the output (that associates with `points`) with `values` (target data).
    If more than one component is provided via a list, the resulting loss will
    be the addative loss of the provided componets.

    Args:
        points: An array of points where the corresponding target values are known and
            used for training.
        values: A scalar or a 2D-array of values that gives the exact solution of the problem.
        component: Integer or a list of integers. The output components satisfying this BC.
            List of integers only supported for the backend PyTorch.
        batch_size: The number of points per minibatch, or `None` to return all points.
            This is only supported for the backend PyTorch and PaddlePaddle.
            Note, If you want to use batch size here, you should also set callback
            'dde.callbacks.PDEPointResampler(bc_points=True)' in training.
        shuffle: Randomize the order on each pass through the data when batching.
    """

    def __init__(self, points, values, component=0, batch_size=None, shuffle=True, scale=1.0, eps=None):
        super().__init__(points, values, component, batch_size, shuffle)
        self.scale = scale
        self.eps = eps

    def error(self, X, inputs, outputs, beg, end, aux_var=None):
        slice_batch = slice(None) if self.batch_size is None else self.batch_indices
        slice_component = slice(self.component, self.component + 1
                                ) if isinstance(self.component, numbers.Number) else self.component
        term_outputs = outputs[beg:end, slice_component] * self.scale
        term_values = self.values[slice_batch] * self.scale
        denominator = 1.0 if self.eps is None else bkd.abs(term_values) + self.eps
        return (term_outputs - term_values) / denominator


class ModifiedPointSetOperatorBC(PointSetOperatorBC):
    """General operator boundary conditions for a set of points.

    Compare the function output, func, (that associates with `points`)
        with `values` (target data).

    Args:
        points: An array of points where the corresponding target values are
            known and used for training.
        values: An array of values which output of function should fulfill.
        func: A function takes arguments (`inputs`, `outputs`, `X`)
            and outputs a tensor of size `N x 1`, where `N` is the length of
            `inputs`. `inputs` and `outputs` are the network input and output
            tensors, respectively; `X` are the NumPy array of the `inputs`.
    """

    def __init__(self, points, values, func, batch_size=None, shuffle=True, scale=1.0, eps=None):
        super().__init__(points, values, func)
        self.batch_size = batch_size
        self.scale = scale
        self.eps = eps

        if batch_size is not None:  # batch iterator and state
            if backend_name not in ["pytorch", "paddle"]:
                raise RuntimeError(
                    "batch_size only implemented for pytorch and paddle backend"
                )
            self.batch_sampler = data.sampler.BatchSampler(len(self), shuffle=shuffle)
            self.batch_indices = None

    def __len__(self):
        return self.points.shape[0]

    def collocation_points(self, X):
        if self.batch_size is not None:
            self.batch_indices = self.batch_sampler.get_next(self.batch_size)
            return self.points[self.batch_indices]
        return self.points

    def error(self, X, inputs, outputs, beg, end, aux_var=None):
        slice_batch = slice(None) if self.batch_size is None else self.batch_indices
        # term_func = self.func(inputs[beg:end], outputs[beg:end], X[beg:end]) * self.scale
        term_func = self.func(inputs, outputs, X)[beg:end] * self.scale
        term_values = self.values[slice_batch] * self.scale
        denominator = 1.0 if self.eps is None else bkd.abs(term_values) + self.eps
        return (term_func - term_values) / denominator
