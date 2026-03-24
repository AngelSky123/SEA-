import torch
import torch.nn as nn
from .encoder import Encoder
from .pose_head import PoseHead
from .domain_disc import DomainDisc
from .disentangle import DomainDisentangle


class WiFiPoseModel(nn.Module):
    def __init__(self, dim=64, num_joints=17):
        super().__init__()
        self.encoder     = Encoder(in_dim=4, dim=dim)
        self.disentangle = DomainDisentangle(dim)
        self.head        = PoseHead(dim, num_joints=num_joints)
        self.domain      = DomainDisc(dim, num_env=2)

    def forward(self, xs, xt, alpha=1.0):
        """
        返回:
          pose      : (B, J, 3)      — 姿态预测（相对坐标）
          vel_pred  : (B, J, 3)      — 预测帧间速度
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

        # PoseHead 现在返回 (pose, vel_pred)
        pose, vel_pred = self.head(fs_shared)

        fs_p_avg = fs_private.mean(dim=(1, 2))
        ft_p_avg = ft_private.mean(dim=(1, 2))
        ds = self.domain(fs_p_avg, alpha)
        dt = self.domain(ft_p_avg, alpha)

        fs_s_avg = fs_shared.mean(dim=(1, 2))
        ft_s_avg = ft_shared.mean(dim=(1, 2))
        orth_s = DomainDisentangle.orthogonality_loss(fs_s_avg, fs_p_avg)
        orth_t = DomainDisentangle.orthogonality_loss(ft_s_avg, ft_p_avg)
        orth_loss = (orth_s + orth_t) * 0.5

        return pose, vel_pred, fs_shared, ft_shared, ds, dt, orth_loss