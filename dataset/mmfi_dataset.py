import os
import numpy as np
import torch
from torch.utils.data import Dataset
import scipy.io as sio


class MMFiDataset(Dataset):
    """
    MMFi CSI 数据集。

    修复（v12）：
      _process_mat 的维度解释完全修正。

      原版假设 .mat shape 为 (Tx发射天线, S接收天线, P子载波)，
      实际 MMFi 的 CSIamp/CSIphase shape 为:

        (Rx接收天线=3, Subcarrier子载波=114, Packet包数=10)

      原版错误地将前两维合并得到 171 个"节点"，每个节点其实是
      (天线, 子载波) 对，语义不合理且计算量大。

      修复后：
        - N = 3（接收天线）= GAT 节点数
        - 对 Packet 维度做均值聚合（每帧 10 个包取平均，获得稳定测量）
        - 子载波维度可选降采样后，展平为每个节点的特征
        - in_dim = n_subcarriers × 4

      这样 GAT 在 3 个天线节点间做注意力，物理意义正确，
      且计算量从 O(171²) 降到 O(3²)。

    改进（v2+）：
      - 支持预处理缓存（cache_dir），消除 CPU IO 瓶颈
      - __getitem__ 返回根节点中心化的 pose

    返回值：(csi, pose_centered, root_offset)
      csi           : (T, N, C)  N=3 天线, C=子载波数×4
      pose_centered : (T, J, 3)  已减去第 0 帧 Hip 坐标
      root_offset   : (3,)       原始根节点坐标
    """

    def __init__(self, root, envs, seq_len=20, cache_dir=None,
                 sub_downsample=2):
        """
        sub_downsample : int
            子载波降采样倍率。
            1 = 不降采样 (114 子载波, in_dim=456)
            2 = 每 2 取 1 (57 子载波, in_dim=228)（默认）
            4 = 每 4 取 1 (28 子载波, in_dim=112)（显存不足时使用）
        """
        self.samples        = []
        self.seq_len        = seq_len
        self.cache_dir      = cache_dir
        self.root           = root
        self.sub_downsample = sub_downsample

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
        # 缓存文件名中编码降采样倍率，避免不同配置的缓存冲突
        base = frame_name.replace(".mat", f"_ds{self.sub_downsample}.npy")
        return os.path.join(cache_subdir, base)

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

    # ── CSI 预处理（v12 修正）────────────────────────────────────────────

    def _process_mat(self, path):
        """
        修正后的 CSI 预处理。

        .mat 文件结构：
          CSIamp:   (Rx=3, Subcarrier=114, Packet=10)
          CSIphase: (Rx=3, Subcarrier=114, Packet=10)

        处理流程：
          1. 提取 amp, phase
          2. Phase unwrap → phase_diff（相邻子载波差分）
          3. 幅度归一化（per-frame）
          4. 构建 4 维特征 [real, imag, amp, phase_diff]
          5. 对 Packet 维度取均值（10 个包 → 1 个稳定测量）
          6. 子载波降采样
          7. 输出 (N=3, S_down, 4)

        返回 shape: (N=3, S_down, 4)
          N      = 3 接收天线（= GAT 节点数）
          S_down = 降采样后的子载波数
          4      = [real, imag, amp, phase_diff]
        """
        mat   = sio.loadmat(path)
        amp   = mat["CSIamp"].astype(np.float32)      # (3, 114, 10)
        phase = mat["CSIphase"].astype(np.float32)     # (3, 114, 10)

        amp   = np.nan_to_num(amp,   nan=0.0, posinf=0.0, neginf=0.0)
        phase = np.nan_to_num(phase, nan=0.0, posinf=0.0, neginf=0.0)

        # Phase unwrap（沿子载波维度 axis=1）
        phase      = np.unwrap(phase, axis=1)
        phase_diff = np.diff(phase, axis=1)
        # 补齐最后一个子载波（复制最后一个差分值）
        phase_diff = np.concatenate([phase_diff, phase_diff[:, -1:, :]], axis=1)
        # phase_diff shape: (3, 114, 10)

        # 幅度归一化（全局 per-frame）
        mean = np.mean(amp)
        std  = max(float(np.std(amp)), 1e-6)
        amp  = (amp - mean) / std

        real = amp * np.cos(phase)
        imag = amp * np.sin(phase)

        # 堆叠 4 维特征
        # shape: (3, 114, 10, 4) = (Rx, Subcarrier, Packet, Feature)
        csi = np.stack([real, imag, amp, phase_diff], axis=-1)

        # 对 Packet 维度取均值 → 每帧一个稳定的 CSI 测量
        # (3, 114, 10, 4) → (3, 114, 4)
        csi = csi.mean(axis=2)

        # 子载波降采样
        # (3, 114, 4) → (3, 114//ds, 4)
        csi = csi[:, ::self.sub_downsample, :]

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
        # csi_seq: (T, N=3, S_down, 4)
        csi_seq = np.stack([
            self.load_csi(sample["csi_dir"], f)
            for f in sample["frames"]
        ])

        # 将 S_down × 4 展平为特征维度 → (T, N=3, S_down*4)
        T, N, S, C = csi_seq.shape
        csi_seq = csi_seq.reshape(T, N, S * C)

        pose = sample["gt"].copy()   # (T, J, 3)

        # ── 根节点中心化 ─────────────────────────────────────────────────
        root_offset = pose[0, 0, :].copy()   # (3,)
        pose        = pose - root_offset     # (T, J, 3)，相对坐标

        return (
            torch.from_numpy(csi_seq.copy()).float(),
            torch.tensor(pose,        dtype=torch.float32),
            torch.tensor(root_offset, dtype=torch.float32),
        )