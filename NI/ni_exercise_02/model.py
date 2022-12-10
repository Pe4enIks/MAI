from torch import nn


class LinearModel(nn.Module):
    def __init__(self, inp, outp):
        super().__init__()
        self.fc = nn.Linear(inp, outp)

    def forward(self, x):
        return self.fc(x)
