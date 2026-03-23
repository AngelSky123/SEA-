import torch.nn as nn
import torch.nn.functional as F
import torch


class DomainDisentangle(nn.Module):
    """
    将 encoder 输出分解为：
      - shared_feat  : 域不变的姿态语义特征（用于姿态预测 & 对齐损失）
      - private_feat : 域私有的环境/噪声特征（用于域分类，不参与姿态预测）
    """

    def __init__(self, dim):
        super().__init__()
        self.shared_head = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.ReLU(),
        )
        self.private_head = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.ReLU(),
        )

    def forward(self, x):
        """
        x : (..., dim)
        返回 (shared, private)，形状与输入相同
        """
        shared = self.shared_head(x)
        private = self.private_head(x)
        return shared, private

    @staticmethod
    def orthogonality_loss(shared, private):
        """
        正交损失：鼓励 shared 和 private 特征不相关。
        输入均为 (B, D) 的平均特征向量。
        """
        # 归一化后计算余弦相似度矩阵的 Frobenius 范数
        s = F.normalize(shared, dim=-1)
        p = F.normalize(private, dim=-1)
        # 理想情况下 s·p^T 应接近零矩阵
        correlation = torch.mm(s, p.T)
        return (correlation ** 2).mean()