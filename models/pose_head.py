import torch
import torch.nn as nn
import torch.nn.functional as F


class PoseHead(nn.Module):
    """
    改进版姿态回归头。

    原版问题：结构过于简单（单层线性），导致模式崩塌——
    模型倾向于输出所有训练样本的均值骨架，MSE 损失无法惩罚这种行为。

    改进：
      1. 加深网络（3层MLP + LayerNorm + Dropout），增强非线性表达能力
      2. 时序特征同时输入首帧和末帧，提供运动方向信息
      3. 输出速度残差（Δpose），从均值姿态做增量预测，避免均值塌陷
    """

    def __init__(self, dim, num_joints=17, dropout=0.1):
        super().__init__()
        self.num_joints = num_joints

        # 空间聚合后的特征维度
        in_dim = dim * 2   # 首帧 + 末帧拼接

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, dim * 2),
            nn.LayerNorm(dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(dim, num_joints * 3),
        )

        # 速度头：预测从首帧到末帧的关节位移
        self.vel_head = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, num_joints * 3),
        )

    def forward(self, x):
        """
        x : (B, T, N, D) — 来自 DomainDisentangle 的共享特征

        返回:
          pose     : (B, J, 3)  — 第 0 帧姿态预测
          vel_pred : (B, J, 3)  — 预测的帧间速度（用于 vel loss 监督）
        """
        B, T, N, D = x.shape

        # 空间维度均值聚合
        x_spa = x.mean(dim=2)          # (B, T, D)

        # 取首帧和末帧特征拼接，提供运动方向信息
        x0 = x_spa[:, 0, :]            # (B, D)  — 起始帧
        xT = x_spa[:, -1, :]           # (B, D)  — 末尾帧
        x_cat = torch.cat([x0, xT], dim=-1)   # (B, 2D)

        pose = self.mlp(x_cat).view(B, self.num_joints, 3)

        # 速度预测：末帧特征 - 首帧特征 → 预测关节位移方向
        vel_pred = self.vel_head(xT - x0).view(B, self.num_joints, 3)

        return pose, vel_pred