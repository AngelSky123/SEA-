import torch.nn as nn
from .gat import GATLayer
from .attention import CrossSensorAttention
from .transformer import TemporalTransformer

class Encoder(nn.Module):
    def __init__(self, in_dim=4, dim=64):  #  改这里
        super().__init__()
        self.gat = GATLayer(in_dim, dim)
        self.attn = CrossSensorAttention(dim)
        self.temp = TemporalTransformer(dim)

    def forward(self, x):
        B,T,N,C = x.shape

        x = x.view(B*T, N, C)
        x = self.gat(x)
        x = self.attn(x)

        x = x.view(B,T,N,-1)
        x = self.temp(x)

        return x