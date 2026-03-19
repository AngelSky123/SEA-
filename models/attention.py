import torch.nn as nn

class CrossSensorAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, 4, batch_first=True)

    def forward(self, x, return_attn=False):
        out, weights = self.attn(x, x, x)

        if return_attn:
            return out, weights
        return out