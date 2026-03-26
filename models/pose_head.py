import torch
import torch.nn as nn


class PoseHead(nn.Module):
    """
    姿态回归头。

    修复（v4）：解耦姿态预测与速度预测。
    原版将 concat([x0, xT]) 输入姿态头，导致末帧特征 xT 把四肢"拉向"
    末帧位置，在动作幅度大的序列（步行、下蹲）中造成预测骨架四肢严重
    扭曲（可视化中腿部出现深红色高误差）。

    修复方案：
      - 姿态头（预测第 0 帧静态姿态）仅使用 x0
      - 速度头（预测首→末帧关节位移）仅使用 (xT - x0)
      两个分支完全解耦，互不干扰。

    注意：修复后姿态头第一层 Linear 的 in_dim 从 dim*2 变回 dim，
    旧 checkpoint 的 head.mlp.0.weight 形状不匹配，加载时会出现
    missing/unexpected key 提示（strict=False 下正常），head 部分
    随机初始化，需重新训练以获得最佳效果。
    """

    def __init__(self, dim, num_joints=17, dropout=0.1):
        super().__init__()
        self.num_joints = num_joints

        # 姿态头：仅用起始帧特征预测第 0 帧静态姿态
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.LayerNorm(dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(dim, num_joints * 3),
        )

        # 速度头：仅用运动差分预测首→末帧关节位移
        self.vel_head = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, num_joints * 3),
        )

    def forward(self, x):
        """
        x : (B, T, N, D) — 来自 DomainDisentangle 的共享特征

        返回:
          pose     : (B, J, 3)  — 第 0 帧姿态预测（仅依赖 x0，与 xT 解耦）
          vel_pred : (B, J, 3)  — 首→末帧关节位移预测（仅依赖 xT-x0）
        """
        B, T, N, D = x.shape

        # 空间维度均值聚合
        x_spa = x.mean(dim=2)        # (B, T, D)

        x0 = x_spa[:, 0, :]         # (B, D) — 起始帧
        xT = x_spa[:, -1, :]        # (B, D) — 末尾帧

        # 修复：姿态头只用 x0，完全不引入末帧信息
        pose = self.mlp(x0).view(B, self.num_joints, 3)

        # 速度头只用运动差分，语义清晰不干扰姿态预测
        vel_pred = self.vel_head(xT - x0).view(B, self.num_joints, 3)

        return pose, vel_pred