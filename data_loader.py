import os
import scipy.io as sio
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class MMFiDataset(Dataset):  
    def __init__(self, root, env='E01', mode='train', num_sensors=342, L=297):
        self.root = root
        self.env = env
        self.mode = mode  
        self.samples = []
        
        env_path = os.path.join(root, env)
        
        # 保护机制：如果路径不存在，生成随机假数据跑通流程
        if not os.path.exists(env_path):
            print(f"⚠️ 警告: 未找到真实路径 {env_path}，正在生成 Dummy Data ({mode})")
            dummy_len = 64 if mode == 'train' else 16
            for _ in range(dummy_len):
                amp = np.random.rand(num_sensors, L)
                pose = np.random.rand(17, 3)
                self.samples.append((amp, pose))
        else:
            # 加载真实数据的逻辑
            for subject in os.listdir(env_path):
                subject_path = os.path.join(env_path, subject)
                if not os.path.isdir(subject_path):
                    continue
                for action in os.listdir(subject_path):
                    csi_path = os.path.join(subject_path, action, 'wifi-csi')
                    # 【修改 1】：将文件名改为 ground_truth.npy
                    pose_path = os.path.join(subject_path, action, 'ground_truth.npy')  
                    
                    if os.path.exists(csi_path) and os.path.exists(pose_path):
                        
                        # ==========================================
                        # 【修改 2】：直接使用 np.load 读取 .npy 文件
                        # ==========================================
                        pose_array = np.load(pose_path)
                        
                        # 处理时间维度：如果标签包含了 297 帧 (形状为 [297, 17, 3])
                        # 我们取这 297 帧的平均骨架作为这一个动作序列的 Ground Truth
                        if pose_array.ndim == 3: 
                            pose = np.mean(pose_array, axis=0) # 变成 [17, 3]
                        else:
                            pose = pose_array # 如果已经是 [17, 3] 则直接使用
                        # ==========================================

                        csi_files = [f for f in os.listdir(csi_path) if f.endswith('.mat')]
                        for f in csi_files:
                            csi = sio.loadmat(os.path.join(csi_path, f))['csi_data']  
                            amp = np.abs(csi)  
                            amp = amp.transpose(1, 2, 0).reshape(3 * 114, 297)  
                            
                            self.samples.append((amp, pose))
                            
            # 分割train/test
            split = int(len(self.samples) * 0.8)
            self.samples = self.samples[:split] if mode == 'train' else self.samples[split:]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        csi, pose = self.samples[idx]
        return torch.from_numpy(csi).float(), torch.from_numpy(pose).float()

def get_loaders(root, source_env='E01', target_env='E02', batch_size=32):
    # drop_last=True 防止最后一个 batch size 为 1 时导致 Deep Coral 计算协方差报错
    source_train = DataLoader(MMFiDataset(root, source_env, 'train'), batch_size, shuffle=True, drop_last=True)
    target_train = DataLoader(MMFiDataset(root, target_env, 'train'), batch_size, shuffle=True, drop_last=True)
    target_test = DataLoader(MMFiDataset(root, target_env, 'test'), batch_size, shuffle=False)
    return source_train, target_train, target_test