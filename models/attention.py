import torch.nn as nn


class CrossSensorAttention(nn.Module):
    """
    跨传感器多头自注意力。

    修复：将注意力权重缓存到实例变量 last_attn_weights，
    而不是通过 return_attn 标志改变返回类型——避免调用方
    意外收到 tuple 导致后续操作出错。
    """

    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.last_attn_weights = None   # 供可视化模块读取

    def forward(self, x):
        """
        x : (B, N, D)
        返回 : (B, N, D)
        """
        out, weights = self.attn(x, x, x)
        self.last_attn_weights = weights.detach()   # 缓存，不影响主路径
        return out