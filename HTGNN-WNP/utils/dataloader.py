import os.path
import torch
from torch.utils.data.dataset import Dataset
from utils.read_json import read_param_data, read_yearlab_data
import time

class waterdataset(Dataset):
    def __init__(self, origin_path, param_path, concentration_path, global_variant, train=True):
        super(waterdataset, self).__init__()
        self.origin_path = origin_path
        self.train = train
        self.param_path = param_path
        self.global_variant = global_variant
        self.lable_path = concentration_path

        if self.train:
            self.globalpath = os.path.join(self.origin_path, "train", self.global_variant)
            self.parampath = os.path.join(self.origin_path, "train", self.param_path)
            self.labpath = os.path.join(self.origin_path, "train", self.lable_path)
        else:
            self.globalpath = os.path.join(self.origin_path, "test", self.global_variant)
            self.parampath = os.path.join(self.origin_path, "test", self.param_path)
            self.labpath = os.path.join(self.origin_path, "test", self.lable_path)

        self.labnames = sorted(os.listdir(self.labpath))

    def __len__(self):
        return len(self.labnames)

    def __getitem__(self, index):
        labname = self.labnames[index]

        node_tensor = read_param_data(os.path.join(self.parampath, r"nodeparam.csv"))
        x_n = torch.narrow(node_tensor, dim=1, start=0, length=3)
        river_tensor = read_param_data(os.path.join(self.parampath, r"riverparam.csv"))
        x_r = torch.narrow(river_tensor, dim=1, start=0, length=12)

        global_tensor = read_yearlab_data(os.path.join(self.globalpath, labname))
        lable_tensor = read_yearlab_data(os.path.join(self.labpath, labname))

        x_g = global_tensor
        y = lable_tensor

        return x_n, x_r, x_g, y


