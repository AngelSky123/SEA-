import torch
import torch.nn as nn


class PoseHead(nn.Module):
    """
    姿态回归头。

    修复（v5）：解决均值塌陷问题。

    塌陷根因：x_spa[:, 0, :] 单帧特征对不同动作几乎相同，
    导致 PoseHead 对所有输入输出几乎相同的均值骨架。

    修复方案（old_style=False 时）：
      pose 输入 = concat(x_max, motion)，其中：
        x_max  = x_spa.max(dim=1).values  ← 全序列时间 max-pooling
                 保留序列中最显著的激活，比单帧 x0 信息量更大
        motion = xT - x0                  ← 首末运动差分
                 直接编码运动方向，是区分不同动作最有效的信号
      pose_in 维度仍为 dim*2，与旧版权重形状兼容（但语义不同，需重训练）

    old_style=True 保留旧版 concat(x0, xT) 结构，用于加载旧 checkpoint。
    """

    def __init__(self, dim, num_joints=17, dropout=0.1, old_style=False):
        super().__init__()
        self.num_joints = num_joints
        self.old_style  = old_style

        pose_in = dim * 2   # 新旧版输入维度均为 dim*2

        self.mlp = nn.Sequential(
            nn.Linear(pose_in, dim * 2),
            nn.LayerNorm(dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(dim, num_joints * 3),
        )

        self.vel_head = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, num_joints * 3),
        )

    def forward(self, x):
        """
        x : (B, T, N, D)

        返回:
          pose     : (B, J, 3)
          vel_pred : (B, J, 3)
        """
        B, T, N, D = x.shape
        x_spa = x.mean(dim=2)          # (B, T, D)
        x0 = x_spa[:, 0, :]           # (B, D)
        xT = x_spa[:, -1, :]          # (B, D)

        if self.old_style:
            # 旧版：concat(x0, xT)，兼容旧 checkpoint
            pose_in = torch.cat([x0, xT], dim=-1)
        else:
            # 新版：全序列 max-pool + 运动差分，信息量显著更大
            x_max   = x_spa.max(dim=1).values          # (B, D)
            motion  = xT - x0                          # (B, D)
            pose_in = torch.cat([x_max, motion], dim=-1)  # (B, 2D)

        pose     = self.mlp(pose_in).view(B, self.num_joints, 3)
        vel_pred = self.vel_head(xT - x0).view(B, self.num_joints, 3)

        return pose, vel_pred