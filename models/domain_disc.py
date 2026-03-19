import torch.nn as nn

class DomainDisc(nn.Module):
    def __init__(self, dim, num_env=4):
        super().__init__()
        self.net = nn.Linear(dim, num_env)

    def forward(self, x):
        return self.net(x)