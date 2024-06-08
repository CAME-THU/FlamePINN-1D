
from deepxde.data.dataset import DataSet
import torch


class ScaledDataSet(DataSet):
    def __init__(
        self,
        X_train=None,
        y_train=None,
        X_test=None,
        y_test=None,
        fname_train=None,
        fname_test=None,
        col_x=None,
        col_y=None,
        standardize=False,
        scales=(1.0, ),
    ):
        super().__init__(X_train, y_train, X_test, y_test, fname_train, fname_test, col_x, col_y, standardize)
        self.scales = torch.tensor(scales)

    def losses(self, targets, outputs, loss_fn, inputs, model, aux=None):
        assert len(self.scales) == targets.shape[1]
        return loss_fn(targets * self.scales, outputs * self.scales)

