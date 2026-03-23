import torch.nn as nn
from .grad_reverse import grad_reverse


class DomainDisc(nn.Module):
    """
    域判别器，内置梯度反转。
    输入特征先经过 GRL（梯度反转），再送入分类头。
    这样 encoder 的梯度方向被反转，促使其学习域不变特征。
    """

    def __init__(self, dim, num_env=2):
        super().__init__()
        # 加深判别器，增强对抗能力
        self.net = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.ReLU(),
            nn.Linear(dim // 2, num_env),
        )

    def forward(self, x, alpha=1.0):
        # 梯度反转：让 encoder 被迫产生域不变特征
        x_rev = grad_reverse(x, alpha)
        return self.net(x_rev)