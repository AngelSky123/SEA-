import torch.nn as nn

class DomainDisentangle(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.shared = nn.Linear(dim, dim)
        self.private = nn.Linear(dim, dim)

    def forward(self, x):
        return self.shared(x), self.private(x)