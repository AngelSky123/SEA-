import os
import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import Dataset, DataLoader

class MMFiDataset(Dataset):  
    def __init__(self, root, envs=['E01'], mode='train', num_sensors=342, L=297):
        self.root = root
        if isinstance(envs, str):
            envs = [envs]
        self.envs = envs
        self.mode = mode  
        self.samples = []
        
        for env in self.envs:
            env_path = os.path.join(root, env)
            
            # 保护机制：如果没有真实数据，生成假数据
            if not os.path.exists(env_path):
                print(f"⚠️ 警告: 未找到真实路径 {env_path}，正在生成 Dummy Data ({mode})")
                dummy_len = 64 if mode == 'train' else 16
                for _ in range(dummy_len):
                    amp = np.random.rand(num_sensors, L)
                    pose = np.random.rand(17, 3)
                    self.samples.append((amp, pose))
                continue
            
            env_samples = []
            
            # 遍历 Subject -> Action -> CSI
            for subject in os.listdir(env_path):
                subject_path = os.path.join(env_path, subject)
                if not os.path.isdir(subject_path):
                    continue
                    
                for action in os.listdir(subject_path):
                    csi_path = os.path.join(subject_path, action, 'wifi-csi')
                    pose_path = os.path.join(subject_path, action, 'ground_truth.npy')  
                    
                    if os.path.exists(csi_path) and os.path.exists(pose_path):
                        # 1. 加载标签
                        pose_array = np.load(pose_path)
                        if pose_array.ndim == 3: 
                            pose = np.mean(pose_array, axis=0)
                        else:
                            pose = pose_array

                        # 2. 读取序列
                        csi_files = sorted([f for f in os.listdir(csi_path) if f.endswith('.mat')])
                        sequence_frames = []
                        
                        for f in csi_files:
                            mat_dict = sio.loadmat(os.path.join(csi_path, f))
                            data_keys = [k for k in mat_dict.keys() if not k.startswith('__')]
                            csi_keys = [k for k in data_keys if 'csi' in k.lower() or 'data' in k.lower()]
                            valid_key = csi_keys[0] if len(csi_keys) > 0 else data_keys[0]
                            
                            csi = mat_dict[valid_key]
                            amp = np.abs(csi)  
                            
                            if amp.size == 3420:
                                frame_feature = amp.reshape(342, -1).mean(axis=1)
                            else:
                                frame_feature = amp.flatten()[:342]
                                
                            sequence_frames.append(frame_feature)
                            
                        # 3. 组装与处理
                        if len(sequence_frames) > 0:
                            seq_amp = np.stack(sequence_frames, axis=1) 
                            
                            L_current = seq_amp.shape[1]
                            L_target = L
                            
                            if L_current > L_target:
                                seq_amp = seq_amp[:, :L_target]
                            elif L_current < L_target:
                                pad_width = ((0, 0), (0, L_target - L_current))
                                seq_amp = np.pad(seq_amp, pad_width, mode='constant', constant_values=0)
                            
                            # ==========================================
                            # 【核心防爆机制】：数据极值清理与归一化
                            # ==========================================
                            # 1. 终极清洗：将所有的 NaN 和 Inf(无穷大) 全部强制转换为 0
                            seq_amp = np.nan_to_num(seq_amp, nan=0.0, posinf=0.0, neginf=0.0)
                            pose = np.nan_to_num(pose, nan=0.0, posinf=0.0, neginf=0.0)
                            
                            # 2. 强制转换为 float32 防止后续运算精度溢出
                            seq_amp = seq_amp.astype(np.float32)
                                
                            # 3. Z-score 归一化
                            seq_amp = (seq_amp - np.mean(seq_amp)) / (np.std(seq_amp) + 1e-5)
                            
                            # 4. 最后一道保险：如果归一化后依然不干净，丢弃
                            if not np.isfinite(seq_amp).all() or not np.isfinite(pose).all():
                                continue
                            
                            # 5. 【修复点】确保干净的数据被装入列表！
                            env_samples.append((seq_amp, pose))
                            
            # ==========================================
            # 【修复点】：UDA 跨环境任务，不再内部切分 80/20，保留 100% 完整环境数据
            # ==========================================
            self.samples.extend(env_samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        csi, pose = self.samples[idx]
        return torch.from_numpy(csi).float(), torch.from_numpy(pose).float()

def get_loaders(root, source_envs=['E01', 'E02', 'E03'], target_env='E04', batch_size=32):
    source_train = DataLoader(MMFiDataset(root, source_envs, 'train'), batch_size, shuffle=True, drop_last=True)
    target_train = DataLoader(MMFiDataset(root, [target_env], 'train'), batch_size, shuffle=True, drop_last=True)
    target_test = DataLoader(MMFiDataset(root, [target_env], 'test'), batch_size, shuffle=False)
    return source_train, target_train, target_test