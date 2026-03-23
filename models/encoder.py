import torch.nn as nn
from .gat import GATLayer
from .attention import CrossSensorAttention
from .transformer import TemporalTransformer


class Encoder(nn.Module):
    """
    三阶段编码器：
      1. GATLayer          — 空间传感器间图注意力
      2. CrossSensorAttention — 跨传感器自注意力
      3. TemporalTransformer  — 时序建模（仅在 T 维度，含位置编码）

    修复：in_dim / dim 均通过参数传入，不再硬编码。
    """

    def __init__(self, in_dim=4, dim=64):
        super().__init__()
        self.gat = GATLayer(in_dim, dim)
        self.attn = CrossSensorAttention(dim)
        self.temp = TemporalTransformer(dim)

    def forward(self, x):
        """
        x : (B, T, N, C)
        返回 : (B, T, N, D)
        """
        B, T, N, C = x.shape

        # 时间帧展平，做空间建模
        x = x.view(B * T, N, C)          # (B*T, N, C)
        x = self.gat(x)                   # (B*T, N, D)
        x = self.attn(x)                  # (B*T, N, D)

        # 恢复时间维度
        x = x.view(B, T, N, -1)           # (B, T, N, D)

        # 时序建模
        x = self.temp(x)                  # (B, T, N, D)

        return x