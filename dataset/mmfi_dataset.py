import os
import numpy as np
import torch
from torch.utils.data import Dataset
import scipy.io as sio


class MMFiDataset(Dataset):
    """
    MMFi CSI 数据集。

    改进：
      - 支持预处理缓存（cache_dir），消除 CPU IO 瓶颈
      - __getitem__ 返回根节点中心化的 pose，消除全局位置偏移问题
        返回值：(csi, pose_centered, root_offset)
        pose_centered : (T, J, 3)，已减去第 0 帧 Hip 坐标
        root_offset   : (3,)，原始根节点坐标，供测试时反归一化使用

    修复（v3）：
      - _process_mat 不再对天线维度做加权求和（原 shape: (P, 4)）。
        加权聚合会把多天线的空间信息在数据预处理阶段丢失，导致后续
        GAT/CrossSensorAttention 无法建模天线间的空间关系。
        修复后保留 (N, P, 4) 形状，N = 天线数（子采样后），P = 子载波数。
        对应地，Encoder 的 in_dim 仍为 4，但输入维度变为 (B, T, N, P*4)
        ——此处保持 P 维度展平后作为特征，让模型自行学习子载波间关系。
        如需兼容旧模型，可将 keep_spatial=False 还原为聚合模式。
    """

    def __init__(self, root, envs, seq_len=20, cache_dir=None,
                 keep_spatial=True):
        """
        keep_spatial : bool
            True  — 保留天线维度，返回 csi shape (T, N, C)，C = subcarrier * 4
            False — 原始加权聚合模式，返回 csi shape (T, 1, C)（向后兼容）
        """
        self.samples      = []
        self.seq_len      = seq_len
        self.cache_dir    = cache_dir
        self.root         = root
        self.keep_spatial = keep_spatial

        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)

        for e in envs:
            e_path = os.path.join(root, e)
            if not os.path.exists(e_path):
                print(f"  [Warning] env path not found: {e_path}")
                continue

            for s in sorted(os.listdir(e_path)):
                s_path = os.path.join(e_path, s)
                if not os.path.isdir(s_path):
                    continue

                for a in sorted(os.listdir(s_path)):
                    a_path  = os.path.join(s_path, a)
                    csi_dir = os.path.join(a_path, "wifi-csi")
                    gt_path = os.path.join(a_path, "ground_truth.npy")

                    if not os.path.exists(csi_dir) or not os.path.exists(gt_path):
                        continue

                    frames = sorted(os.listdir(csi_dir))
                    gt     = np.load(gt_path)   # (F, J, 3)

                    for i in range(len(frames) - seq_len):
                        self.samples.append({
                            "frames":  frames[i : i + seq_len],
                            "csi_dir": csi_dir,
                            "gt":      gt[i : i + seq_len],
                        })

        print(f"  MMFiDataset: {len(self.samples)} samples from envs {envs}")
        if cache_dir:
            print(f"  Cache dir  : {cache_dir}")
            self._prebuild_cache()

    # ── 缓存 ─────────────────────────────────────────────────────────────

    def _cache_path(self, csi_dir, frame_name):
        rel          = os.path.relpath(csi_dir, start=self.root)
        cache_subdir = os.path.join(self.cache_dir, rel)
        os.makedirs(cache_subdir, exist_ok=True)
        return os.path.join(cache_subdir, frame_name.replace(".mat", ".npy"))

    def _prebuild_cache(self):
        all_frames = set()
        for s in self.samples:
            for f in s["frames"]:
                all_frames.add((s["csi_dir"], f))

        missing = [
            (d, f) for d, f in all_frames
            if not os.path.exists(self._cache_path(d, f))
        ]

        if not missing:
            print(f"  Cache complete ({len(all_frames)} frames).")
            return

        print(f"  Building cache: {len(missing)}/{len(all_frames)} frames ...")
        try:
            from tqdm import tqdm
            iterator = tqdm(missing, ncols=80, unit="frame")
        except ImportError:
            iterator = missing

        for i, (csi_dir, fname) in enumerate(iterator):
            csi = self._process_mat(os.path.join(csi_dir, fname))
            np.save(self._cache_path(csi_dir, fname), csi.astype(np.float32))
            if not hasattr(iterator, "update") and i % 500 == 0:
                print(f"    {i}/{len(missing)} frames cached ...")

        print("  Cache build complete.")

    # ── CSI 预处理 ───────────────────────────────────────────────────────

    def _process_mat(self, path):
        """
        返回 shape: (N, P, 4)
          N = 天线数（降采样后）
          P = 子载波数（降采样后）
          4 = [real, imag, amp, phase_diff]

        修复：不再对天线做加权求和，保留空间维度供模型学习。
        """
        mat   = sio.loadmat(path)
        amp   = mat["CSIamp"].astype(np.float32)
        phase = mat["CSIphase"].astype(np.float32)

        amp   = np.nan_to_num(amp,   nan=0.0, posinf=0.0, neginf=0.0)
        phase = np.nan_to_num(phase, nan=0.0, posinf=0.0, neginf=0.0)

        phase      = np.unwrap(phase, axis=-1)
        phase_diff = np.diff(phase, axis=-1)
        phase_diff = np.concatenate([phase_diff, phase_diff[..., -1:]], axis=-1)

        # 幅度归一化
        mean = np.mean(amp)
        std  = max(float(np.std(amp)), 1e-6)
        amp  = (amp - mean) / std

        real = amp * np.cos(phase)
        imag = amp * np.sin(phase)

        # csi: (Tx, S, P, 4)
        csi = np.stack([real, imag, amp, phase_diff], axis=-1)

        # 天线降采样：每隔 2 取 1
        csi = csi[:, ::2, :, :]          # (Tx, N, P, 4)

        # 合并 Tx 维度到 N（多根发射天线视为多个节点）
        Tx, N, P, C = csi.shape
        csi = csi.reshape(Tx * N, P, C)  # (N_total, P, 4)

        return np.nan_to_num(csi).astype(np.float32)

    def load_csi(self, csi_dir, frame_name):
        if self.cache_dir:
            cp = self._cache_path(csi_dir, frame_name)
            if os.path.exists(cp):
                return np.load(cp)
        return self._process_mat(os.path.join(csi_dir, frame_name))

    # ── Dataset 接口 ─────────────────────────────────────────────────────

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample  = self.samples[idx]
        # csi_seq: (T, N, P, 4)
        csi_seq = np.stack([
            self.load_csi(sample["csi_dir"], f)
            for f in sample["frames"]
        ])

        if not self.keep_spatial:
            # 向后兼容：对 N 维度加权聚合 → (T, P, 4)，再 unsqueeze → (T, 1, P, 4)
            node_var = np.var(csi_seq, axis=(2, 3))        # (T, N)
            node_w   = node_var / (node_var.sum(axis=1, keepdims=True) + 1e-6)
            csi_seq  = (csi_seq * node_w[:, :, None, None]).sum(axis=1, keepdims=True)

        # 将 P*4 展平为特征维度 → (T, N, P*4)
        T, N, P, C = csi_seq.shape
        csi_seq = csi_seq.reshape(T, N, P * C)

        pose = sample["gt"].copy()   # (T, J, 3)

        # ── 根节点中心化 ─────────────────────────────────────────────────
        root_offset = pose[0, 0, :].copy()   # (3,)
        pose        = pose - root_offset     # (T, J, 3)，相对坐标

        return (
            torch.from_numpy(csi_seq.copy()).float(),
            torch.tensor(pose,        dtype=torch.float32),
            torch.tensor(root_offset, dtype=torch.float32),
        )