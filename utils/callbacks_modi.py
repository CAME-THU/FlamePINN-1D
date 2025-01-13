
import torch
from deepxde import callbacks
# from deepxde.backend import backend_name
# from deepxde import utils
import csv


class VariableSaver(callbacks.Callback):
    """Monitor and save the learning history of external trainable variables."""

    def __init__(self, var_dict, scale_dict, period=1, filename=None):
        super().__init__()
        self.var_dict = var_dict
        self.scale_dict = scale_dict
        self.period = period
        self.filename = filename

        self.value_dict = {}
        self.value_history = []
        self.epochs_since_last = 0

        csvfile = open(self.filename, "a", newline="")
        writer = csv.writer(csvfile)
        writer.writerow(["epoch"] + list(var_dict.keys()))
        csvfile.close()

    def on_train_begin(self):
        for k, v in self.var_dict.items():
            self.value_dict[k] = v.detach().item() / self.scale_dict[k]

        row = [self.model.train_state.epoch] + list(self.value_dict.values())
        self.value_history.append(row)
        row_formatted = [f"{item:.4e}" for item in row]

        csvfile = open(self.filename, "a", newline="")
        writer = csv.writer(csvfile)
        writer.writerow(row_formatted)
        csvfile.close()

    def on_epoch_end(self):
        self.epochs_since_last += 1
        if self.epochs_since_last >= self.period:
            self.epochs_since_last = 0
            self.on_train_begin()

    def on_train_end(self):
        if not self.epochs_since_last == 0:
            self.on_train_begin()


class ThetaSaver(callbacks.Callback):
    """记录指定 (x, t) 点的 theta 值变化，并保存到 CSV 文件中。"""

    def __init__(self, maps_instance, specific_points, scale, period=1, filename="theta_values.csv"):
        super().__init__()
        self.maps_instance = maps_instance
        self.specific_points_tensor = torch.tensor(specific_points, dtype=torch.float32)
        self.scale = scale
        self.period = period
        self.filename = filename
        self.epochs_since_last = 0

        # 创建 CSV 文件，写入表头
        with open(self.filename, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["epoch"] + [f"theta_{pt}" for pt in specific_points])

    def on_epoch_end(self):
        """每隔指定步数记录一次 theta 值"""
        self.epochs_since_last += 1
        if self.epochs_since_last >= self.period:
            self.epochs_since_last = 0  # 重置计数器

            # 计算 theta 值
            theta_values = self.maps_instance.compute_theta(self.specific_points_tensor) / self.scale
            epoch = self.model.train_state.epoch

            # 将当前 epoch 和 theta 值保存到 CSV
            row = [epoch] + theta_values.tolist()
            with open(self.filename, "a", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(row)

