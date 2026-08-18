import torch.nn as nn


class Expert(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.gelu = nn.GELU(approximate="tanh")
        self.f_u = nn.Linear(input_dim, hidden_dim)
        self.f_d = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        return self.f_d(self.gelu(self.f_u(x)))
