import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
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
        self.register_buffer("pe", pe.unsqueeze(0))   # (1, max_len, dim)

    def forward(self, x):
        # x: (B, T, D)
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class TemporalTransformer(nn.Module):
    """
    时序 Transformer，含正弦位置编码。

    修复：原版在时序建模前先对 N 维度做 mean(dim=2)，再将结果 expand
    回 (B, T, N, D)。这意味着所有节点在经过 Transformer 后特征完全相同，
    CrossSensorAttention 在此之后无法区分节点差异。

    修复方案：将 (B, T, N, D) reshape 为 (B*N, T, D)，让每个节点
    独立经过时序建模，保留节点间的差异性。
    """

    def __init__(self, dim=64, nhead=4, num_layers=3, dropout=0.1):
        super().__init__()
        self.pos_enc = PositionalEncoding(dim, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=nhead,
            dim_feedforward=dim * 8,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        """
        x : (B, T, N, D)
        返回 : (B, T, N, D)，每个节点的时序特征独立建模
        """
        B, T, N, D = x.shape

        # 修复：(B, T, N, D) → (B*N, T, D)，每个节点独立走时序 Transformer
        x_r = x.permute(0, 2, 1, 3).reshape(B * N, T, D)   # (B*N, T, D)
        x_r = self.pos_enc(x_r)
        x_r = self.encoder(x_r)                              # (B*N, T, D)

        # 还原为 (B, T, N, D)
        x_out = x_r.reshape(B, N, T, D).permute(0, 2, 1, 3)
        return x_out