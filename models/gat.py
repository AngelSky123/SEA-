import torch
import torch.nn as nn

class GATLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)
        self.attn = nn.Linear(2*out_dim, 1)

    def forward(self, x):
        h = self.fc(x)
        N = h.size(1)

        h1 = h.unsqueeze(2).repeat(1,1,N,1)
        h2 = h.unsqueeze(1).repeat(1,N,1,1)

        e = self.attn(torch.cat([h1,h2], dim=-1)).squeeze(-1)
        alpha = torch.softmax(e, dim=-1)

        return torch.matmul(alpha, h)