import torch.nn as nn

class TemporalTransformer(nn.Module):
    def __init__(self, dim=64):
        super().__init__()
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=dim, nhead=4),
            num_layers=2
        )

    def forward(self, x):
        B, T, N, D = x.shape

        # ❗ 关键修复：用 reshape
        x = x.reshape(B, T * N, D)

        x = self.encoder(x)

        x = x.reshape(B, T, N, D)

        return x