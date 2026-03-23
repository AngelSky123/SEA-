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
    """

    def __init__(self, root, envs, seq_len=20, cache_dir=None):
        self.samples   = []
        self.seq_len   = seq_len
        self.cache_dir = cache_dir
        self.root      = root

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
                            "frames":  frames[i:i + seq_len],
                            "csi_dir": csi_dir,
                            "gt":      gt[i:i + seq_len],
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
        return os.path.join(cache_subdir, frame_name.replace('.mat', '.npy'))

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
            iterator = tqdm(missing, ncols=80, unit='frame')
        except ImportError:
            iterator = missing

        for i, (csi_dir, fname) in enumerate(iterator):
            csi = self._process_mat(os.path.join(csi_dir, fname))
            np.save(self._cache_path(csi_dir, fname), csi.astype(np.float32))
            if not hasattr(iterator, 'update') and i % 500 == 0:
                print(f"    {i}/{len(missing)} frames cached ...")

        print("  Cache build complete.")

    # ── CSI 预处理 ───────────────────────────────────────────────────────

    def _process_mat(self, path):
        mat   = sio.loadmat(path)
        amp   = mat['CSIamp'].astype(np.float32)
        phase = mat['CSIphase'].astype(np.float32)

        amp   = np.nan_to_num(amp,   nan=0.0, posinf=0.0, neginf=0.0)
        phase = np.nan_to_num(phase, nan=0.0, posinf=0.0, neginf=0.0)

        phase      = np.unwrap(phase, axis=-1)
        phase_diff = np.diff(phase, axis=-1)
        phase_diff = np.concatenate([phase_diff, phase_diff[..., -1:]], axis=-1)

        mean = np.mean(amp)
        std  = max(float(np.std(amp)), 1e-6)
        amp  = (amp - mean) / std

        real = amp * np.cos(phase)
        imag = amp * np.sin(phase)

        csi = np.stack([real, imag, amp, phase_diff], axis=-1)  # (Tx, S, P, 4)
        csi = csi[:, ::2, :, :]
        csi = csi.reshape(-1, csi.shape[2], 4)                  # (N, P, 4)

        node_var = np.var(csi, axis=(1, 2))
        node_w   = node_var / (node_var.sum() + 1e-6)
        csi      = (csi * node_w[:, None, None]).sum(axis=0)    # (P, 4)

        return np.nan_to_num(csi)

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
        csi_seq = np.stack([
            self.load_csi(sample["csi_dir"], f)
            for f in sample["frames"]
        ])                              # (T, P, 4)

        pose = sample["gt"].copy()     # (T, J, 3)

        # ── 根节点中心化（关键改进）─────────────────────────────────────
        # 用第 0 帧 Hip（joint index=0）作为坐标原点
        # 消除绝对世界坐标带来的全局位置偏移问题
        root_offset = pose[0, 0, :].copy()   # (3,)
        pose        = pose - root_offset     # (T, J, 3)，相对坐标

        return (
            torch.from_numpy(csi_seq.copy()),
            torch.tensor(pose,        dtype=torch.float32),
            torch.tensor(root_offset, dtype=torch.float32),
        )