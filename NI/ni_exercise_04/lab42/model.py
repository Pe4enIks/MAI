from torch import nn
import torch


class RBF(nn.Module):
    def __init__(self, rbf_features, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.rbf_features = rbf_features
        self.out_features = out_features
        self.centres = nn.Parameter(torch.Tensor(rbf_features, in_features))
        self.sigmas = nn.Parameter(torch.Tensor(rbf_features))
        self.sw = nn.Linear(rbf_features, rbf_features)
        self.linear = nn.Linear(rbf_features, out_features)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.centres, 0, 1)
        nn.init.constant_(self.sigmas, 0)

    def forward(self, input):
        size = (input.size(0), self.rbf_features, self.in_features)
        x = input.unsqueeze(1).expand(size)
        c = self.centres.unsqueeze(0).expand(size)
        exp = torch.exp(self.sigmas).unsqueeze(0)
        l2 = (x - c).pow(2).sum(-1).pow(0.5)
        distances = l2 / exp
        out = self.sw(distances)
        out = self.linear(out)
        return out
