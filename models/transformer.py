import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, dim, max_len=200, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class TemporalTransformer(nn.Module):
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
        B, T, N, D = x.shape
        x_temp = x.mean(dim=2)
        x_temp = self.pos_enc(x_temp)
        x_temp = self.encoder(x_temp)
        return x_temp.unsqueeze(2).expand(-1, -1, N, -1)
