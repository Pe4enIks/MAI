from torch import nn


class Perceptron_Pytorch(nn.Module):
    def __init__(self, inp, outp):
        super().__init__()
        self.fc = nn.Linear(inp, outp)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.fc(x))
