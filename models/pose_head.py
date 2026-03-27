import torch
import torch.nn as nn


class PoseHead(nn.Module):
    """
    姿态回归头。

    修复（v4）：解耦姿态预测与速度预测，新增 old_style 参数向后兼容。

    old_style=True  （旧版，对应已有 checkpoint）：
        姿态头输入 concat([x0, xT])，in_dim = dim*2
        — 保留此结构是为了让旧 checkpoint 能直接加载，评估 Bug1 修复效果

    old_style=False （新版，重训练后使用）：
        姿态头输入仅 x0，in_dim = dim
        — 消除末帧特征对第 0 帧预测的干扰，修复四肢扭曲问题
        — 速度头输入仅 (xT - x0)，两支路完全解耦

    新训练时默认使用 old_style=False。
    """

    def __init__(self, dim, num_joints=17, dropout=0.1, old_style=False):
        super().__init__()
        self.num_joints = num_joints
        self.old_style  = old_style

        # 根据结构选择姿态头输入维度
        pose_in = dim * 2 if old_style else dim

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
        x_spa = x.mean(dim=2)        # (B, T, D)
        x0 = x_spa[:, 0, :]         # (B, D)
        xT = x_spa[:, -1, :]        # (B, D)

        if self.old_style:
            # 旧版：concat 输入，兼容旧 checkpoint
            pose_in = torch.cat([x0, xT], dim=-1)   # (B, 2D)
        else:
            # 新版：仅用 x0，消除末帧干扰
            pose_in = x0                             # (B, D)

        pose     = self.mlp(pose_in).view(B, self.num_joints, 3)
        vel_pred = self.vel_head(xT - x0).view(B, self.num_joints, 3)

        return pose, vel_pred