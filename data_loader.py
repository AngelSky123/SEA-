import os
import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import Dataset, DataLoader

class MMFiDataset(Dataset):  
    # 【修改 1】：参数改为 envs，支持传入列表（如 ['E01', 'E02', 'E03']）
    def __init__(self, root, envs=['E01'], mode='train', num_sensors=342, L=297):
        self.root = root
        # 如果传入的是单个字符串，统一转成列表以方便遍历
        if isinstance(envs, str):
            envs = [envs]
        self.envs = envs
        self.mode = mode  
        self.samples = []
        
        # 【修改 2】：遍历所有的环境
        for env in self.envs:
            env_path = os.path.join(root, env)
            
            if not os.path.exists(env_path):
                print(f"⚠️ 警告: 未找到真实路径 {env_path}，正在生成 Dummy Data ({mode})")
                dummy_len = 64 if mode == 'train' else 16
                for _ in range(dummy_len):
                    amp = np.random.rand(num_sensors, L)
                    pose = np.random.rand(17, 3)
                    self.samples.append((amp, pose))
                continue
            
            # 用于暂存当前单个环境的数据
            env_samples = []
            for subject in os.listdir(env_path):
                subject_path = os.path.join(env_path, subject)
                if not os.path.isdir(subject_path):
                    continue
                for action in os.listdir(subject_path):
                    csi_path = os.path.join(subject_path, action, 'wifi-csi')
                    pose_path = os.path.join(subject_path, action, 'ground_truth.npy')  
                    
                    if os.path.exists(csi_path) and os.path.exists(pose_path):
                        pose_array = np.load(pose_path)
                        if pose_array.ndim == 3: 
                            pose = np.mean(pose_array, axis=0) # [17, 3]
                        else:
                            pose = pose_array

                        csi_files = [f for f in os.listdir(csi_path) if f.endswith('.mat')]
                        for f in csi_files:
                            csi = sio.loadmat(os.path.join(csi_path, f))['csi_data']  
                            amp = np.abs(csi)  
                            amp = amp.transpose(1, 2, 0).reshape(3 * 114, 297)  
                            
                            env_samples.append((amp, pose))
                            
            # 【修改 3】：针对每个环境独立划分 80% 训练集和 20% 测试集，然后再汇总
            split = int(len(env_samples) * 0.8)
            if mode == 'train':
                self.samples.extend(env_samples[:split])
            else:
                self.samples.extend(env_samples[split:])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        csi, pose = self.samples[idx]
        return torch.from_numpy(csi).float(), torch.from_numpy(pose).float()

# 【修改 4】：get_loaders 支持 source_envs 列表输入
def get_loaders(root, source_envs=['E01', 'E02', 'E03'], target_env='E04', batch_size=32):
    # 源域包含多个环境
    source_train = DataLoader(MMFiDataset(root, source_envs, 'train'), batch_size, shuffle=True, drop_last=True)
    # 目标域只包含测试环境
    target_train = DataLoader(MMFiDataset(root, [target_env], 'train'), batch_size, shuffle=True, drop_last=True)
    target_test = DataLoader(MMFiDataset(root, [target_env], 'test'), batch_size, shuffle=False)
    return source_train, target_train, target_test