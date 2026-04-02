import torch.nn as nn


class CrossSensorAttention(nn.Module):
    """
    跨传感器多头自注意力。

    修复（v6）：新增残差连接 + LayerNorm。
    原版 forward 直接返回 attention 输出，丢弃了输入信息。
    当注意力权重趋于均匀时，所有节点被平均为相同向量，
    导致 Encoder 特征塌陷（诊断 cos_sim=0.9998）。
    残差连接确保输入信息不被完全覆盖。
    """

    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.last_attn_weights = None

    def forward(self, x):
        """
        x : (B, N, D)
        返回 : (B, N, D)
        """
        out, weights = self.attn(x, x, x)
        self.last_attn_weights = weights.detach()
        return self.norm(x + out)      # 残差 + LayerNorm
