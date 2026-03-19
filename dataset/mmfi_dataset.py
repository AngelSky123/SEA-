import os
import numpy as np
import torch
from torch.utils.data import Dataset
import scipy.io as sio

class MMFiDataset(Dataset):
    def __init__(self, root, envs, seq_len=20):
        self.samples = []
        self.seq_len = seq_len

        for e in envs:
            e_path = os.path.join(root, e)
            if not os.path.exists(e_path):
                continue

            for s in os.listdir(e_path):
                s_path = os.path.join(e_path, s)

                for a in os.listdir(s_path):
                    a_path = os.path.join(s_path, a)

                    csi_dir = os.path.join(a_path, "wifi-csi")
                    gt_path = os.path.join(a_path, "ground_truth.npy")

                    if not os.path.exists(csi_dir):
                        continue

                    frames = sorted(os.listdir(csi_dir))
                    gt = np.load(gt_path)

                    for i in range(len(frames) - seq_len):
                        self.samples.append({
                            "frames": frames[i:i+seq_len],
                            "csi_dir": csi_dir,
                            "gt": gt[i:i+seq_len],
                        })

    def __len__(self):
        return len(self.samples)

    def load_csi(self, path):
        mat = sio.loadmat(path)

        amp = mat['CSIamp']
        phase = mat['CSIphase']

        # ===== NaN/Inf 清理 =====
        amp = np.nan_to_num(amp, nan=0.0, posinf=0.0, neginf=0.0)
        phase = np.nan_to_num(phase, nan=0.0, posinf=0.0, neginf=0.0)

        # ===== phase unwrap =====
        phase = np.unwrap(phase, axis=-1)

        # ===== phase diff =====
        phase_diff = np.diff(phase, axis=-1)
        phase_diff = np.concatenate([phase_diff, phase_diff[..., -1:]], axis=-1)

        # ===== amplitude 标准化 =====
        mean = np.mean(amp)
        std = np.std(amp)
        if std < 1e-6:
            std = 1e-6
        amp = (amp - mean) / std

        # ===== 复数建模 =====
        real = amp * np.cos(phase)
        imag = amp * np.sin(phase)

        # ===== 多通道拼接 =====
        csi = np.stack([real, imag, amp, phase_diff], axis=-1)
        # (3,114,10,4)

        Tx, S, P, C = csi.shape

        # ===== 降采样（减少显存）=====
        csi = csi[:, ::2, :, :]   # 114 -> 57

        # ===== reshape =====
        csi = csi.reshape(Tx * (S//2), P, C)  # (171,10,4)

        # ===== packet attention =====
        weights = np.var(csi, axis=2)
        weights = np.exp(weights) / (np.sum(np.exp(weights), axis=1, keepdims=True) + 1e-6)

        csi = (csi * weights[..., None]).sum(axis=1)  # (171,4)

        # ===== 最终保险 =====
        if np.isnan(csi).any():
            csi = np.nan_to_num(csi)

        return csi

    def __getitem__(self, idx):
        sample = self.samples[idx]

        csi_seq = []
        for f in sample["frames"]:
            csi = self.load_csi(os.path.join(sample["csi_dir"], f))
            csi_seq.append(csi)

        csi_seq = np.stack(csi_seq)  # (T,171,4)
        pose = sample["gt"]

        return torch.tensor(csi_seq, dtype=torch.float32), \
               torch.tensor(pose, dtype=torch.float32)