import torch
import torch.nn as nn
from .encoder import Encoder
from .pose_head import PoseHead
from .domain_disc import DomainDisc
from .disentangle import DomainDisentangle


class WiFiPoseModel(nn.Module):
    def __init__(self, in_dim=40, dim=64, num_joints=17):
        """
        修复（v4）：in_dim 不再硬编码为 40，改为外部传入参数。

        实际值由调用方从数据集探测后传入：
            sample_csi, _, _ = dataset[0]
            in_dim = sample_csi.shape[-1]   # (T, N, P*C) 最后一维
            model  = WiFiPoseModel(in_dim=in_dim, dim=cfg.model.dim, ...)

        默认值保留 40 以兼容旧代码路径，但强烈建议显式探测传入。
        """
        super().__init__()
        self.encoder     = Encoder(in_dim=in_dim, dim=dim)
        self.disentangle = DomainDisentangle(dim)
        self.head        = PoseHead(dim, num_joints=num_joints)
        self.domain      = DomainDisc(dim, num_env=2)

    def forward(self, xs, xt, alpha=1.0):
        """
        返回:
          pose      : (B, J, 3)      — 姿态预测（相对坐标，仅依赖起始帧特征）
          vel_pred  : (B, J, 3)      — 首→末帧关节位移预测
          fs_shared : (B, T, N, D)
          ft_shared : (B, T, N, D)
          ds        : (B, 2)
          dt        : (B, 2)
          orth_loss : scalar
        """
        fs_raw = self.encoder(xs)
        ft_raw = self.encoder(xt)

        fs_shared, fs_private = self.disentangle(fs_raw)
        ft_shared, ft_private = self.disentangle(ft_raw)

        pose, vel_pred = self.head(fs_shared)

        # 域判别器作用于 shared 特征（经 GRL 梯度反转）
        fs_s_avg = fs_shared.mean(dim=(1, 2))   # (B, D)
        ft_s_avg = ft_shared.mean(dim=(1, 2))   # (B, D)
        ds = self.domain(fs_s_avg, alpha)
        dt = self.domain(ft_s_avg, alpha)

        # 正交损失：shared ⊥ private
        fs_p_avg = fs_private.mean(dim=(1, 2))
        ft_p_avg = ft_private.mean(dim=(1, 2))
        orth_s = DomainDisentangle.orthogonality_loss(fs_s_avg, fs_p_avg)
        orth_t = DomainDisentangle.orthogonality_loss(ft_s_avg, ft_p_avg)
        orth_loss = (orth_s + orth_t) * 0.5

        return pose, vel_pred, fs_shared, ft_shared, ds, dt, orth_loss