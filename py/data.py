import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import torch.utils.data as Data


class MyData:
    def __init__(self):

        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

        x = np.load("dataset/mix_dos_feature_dset_104343.npz",)['arr_0']
        y = np.load("dataset/mix_dos_label_dset_104343.npz")['arr_0']

        train_size = int(len(x)*0.64)
        valid_size = int(len(x)*0.80)
        test_size = int(len(x)*0.2)


        # 把数据都放入 设备中 GPU/CPU
        self.x = torch.unsqueeze(torch.Tensor(x), dim=1).float().to(DEVICE)
        self.y = torch.Tensor(y).long().to(DEVICE)

        train_data = TensorDataset(self.x[:train_size], self.y[:train_size])
        self.train_loader = Data.DataLoader(
            dataset=train_data,  # 训练的数据
            batch_size=64,
            shuffle=False,  # 不打乱
            num_workers=2,
        )

        self.x_valid_data = self.x[train_size:valid_size]
        self.y_valid_data = self.y[train_size:valid_size]