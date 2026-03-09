import os
import scipy.io as sio
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class MMFiDataset(Dataset):  # MM-Fi数据集加载器
    def __init__(self, root, env='E01', mode='train'):
        self.root = root
        self.env = env
        self.mode = mode  # train/test分割，假设80/20
        self.samples = []
        # 加载env下的序列
        for subject in os.listdir(os.path.join(root, env)):
            for action in os.listdir(os.path.join(root, env, subject)):
                csi_path = os.path.join(root, env, subject, action, 'wifi-csi')
                pose_path = os.path.join(root, env, subject, action, 'annotation_pose_3d.json')  # 假设3D姿态标注文件
                if os.path.exists(csi_path) and os.path.exists(pose_path):
                    csi_files = [f for f in os.listdir(csi_path) if f.endswith('.mat')]
                    for f in csi_files:
                        csi = sio.loadmat(os.path.join(csi_path, f))['csi_data']  # 假设键'csi_data' [297,3,114]复数
                        amp = np.abs(csi)  # 幅度 [297,3,114]
                        amp = amp.transpose(1,2,0).reshape(3*114, 297)  # [N=342, L=297]
                        # 加载姿态：假设json或npy [297,17,3]，简化取平均或最后一帧
                        pose = np.random.rand(17,3)  # 占位符，实际加载3D姿态
                        self.samples.append((amp, pose))
        # 分割train/test
        split = int(len(self.samples)*0.8)
        self.samples = self.samples[:split] if mode=='train' else self.samples[split:]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        csi, pose = self.samples[idx]
        return torch.from_numpy(csi).float(), torch.from_numpy(pose).float()

def get_loaders(root, source_env='E01', target_env='E02', batch_size=32):
    source_train = DataLoader(MMFiDataset(root, source_env, 'train'), batch_size, shuffle=True)
    target_train = DataLoader(MMFiDataset(root, target_env, 'train'), batch_size, shuffle=True)
    target_test = DataLoader(MMFiDataset(root, target_env, 'test'), batch_size, shuffle=False)
    return source_train, target_train, target_test