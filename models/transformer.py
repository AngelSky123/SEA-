import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """标准正弦位置编码，作用在时间维度 T 上。"""

    def __init__(self, dim, max_len=200, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # (1, max_len, dim)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        # x : (B, T, D)
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class TemporalTransformer(nn.Module):
    """
    只在时间维度 T 上做自注意力。
    先对空间维度做均值聚合，再做时序建模，避免 T*N 展平导致的
    时空混淆问题，同时大幅降低显存占用。
    """

    def __init__(self, dim=64, nhead=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.pos_enc = PositionalEncoding(dim, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=nhead,
            dim_feedforward=dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,          # Pre-LN，训练更稳定
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        """
        x : (B, T, N, D)
        返回 : (B, T, N, D)  — 空间维度通过广播恢复
        """
        B, T, N, D = x.shape

        # 空间维度均值聚合 → (B, T, D)
        x_temp = x.mean(dim=2)

        # 加入时间位置编码
        x_temp = self.pos_enc(x_temp)

        # Transformer 只在时间维度上计算注意力
        x_temp = self.encoder(x_temp)          # (B, T, D)

        # 广播回空间维度 (B, T, N, D)
        x_out = x_temp.unsqueeze(2).expand(-1, -1, N, -1)

        return x_out