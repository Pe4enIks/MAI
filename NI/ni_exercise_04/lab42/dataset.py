from torch.utils.data import Dataset as Base
import torch


class Dataset(Base):
    def __init__(self, x, func):
        super().__init__()
        self.x = torch.FloatTensor(x)
        self.y = torch.FloatTensor([func(el) for el in x])

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]
