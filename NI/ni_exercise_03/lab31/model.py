from torch import nn


class MLP(nn.Module):
    def __init__(self, inp, outp):
        super().__init__()
        self.fc1 = nn.Linear(inp, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, outp)
        self.act = nn.LeakyReLU(negative_slope=0.05)

    def forward(self, x):
        h1 = self.act(self.fc1(x))
        h2 = self.act(self.fc2(h1))
        out = self.fc3(h2)
        return out
