import torch
import torch.nn as nn

class GATLayer(nn.Module):
    """
    用线性投影 + 多头自注意力替代原始 GAT 的 O(N²) 手写实现。
    参数量更大，计算效率更高，GPU 利用率更好。
    """
    def __init__(self, in_dim, out_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)
        self.attn = nn.MultiheadAttention(
            out_dim, num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.ffn = nn.Sequential(
            nn.Linear(out_dim, out_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim * 4, out_dim),
        )
        self.norm1 = nn.LayerNorm(out_dim)
        self.norm2 = nn.LayerNorm(out_dim)

    def forward(self, x):
        h = self.proj(x)
        attn_out, _ = self.attn(h, h, h)
        h = self.norm1(h + attn_out)
        h = self.norm2(h + self.ffn(h))
        return h
