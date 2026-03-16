import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MSGC(nn.Module):
    def __init__(self, d_model, num_branches=3):
        super(MSGC, self).__init__()
        self.num_branches = num_branches
        self.queries = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(num_branches)])
        self.keys = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(num_branches)])

    def forward(self, Z_T):
        E_T_branches = []
        for i in range(self.num_branches):
            Q = self.queries[i](Z_T)
            K = self.keys[i](Z_T)
            attn = torch.matmul(Q, K.transpose(1, 2)) / math.sqrt(Z_T.size(-1))
            E_i = F.softmax(attn, dim=-1)
            E_T_branches.append(E_i)
        E_T = torch.mean(torch.stack(E_T_branches), dim=0)
        return E_T

class MPNN(nn.Module):
    def __init__(self, d_model):
        super(MPNN, self).__init__()
        self.mlp = nn.Linear(d_model, d_model)

    def forward(self, Z_T, E_T):
        h = torch.matmul(E_T, Z_T)
        Z_T_updated = F.relu(self.mlp(h))
        return Z_T_updated

class GraphEncoder(nn.Module):
    def __init__(self, num_sensors, d_model, num_branches=3, lstm_layers=2):
        super(GraphEncoder, self).__init__()
        self.msgc = MSGC(d_model, num_branches)
        self.mpnn = MPNN(d_model)
        self.lstm = nn.LSTM(d_model, d_model, lstm_layers, batch_first=True, bidirectional=False)

    def forward(self, patches):
        B, hat_L, N, d = patches.shape
        Z_seq, E_seq = [], []
        
        for T in range(hat_L):
            Z_T = patches[:, T]
            E_T = self.msgc(Z_T)
            Z_T = self.mpnn(Z_T, E_T)
            Z_seq.append(Z_T)
            E_seq.append(E_T)
            
        Z_seq = torch.stack(Z_seq, dim=1)
        E_seq = torch.stack(E_seq, dim=1)
        
        Z_seq_resh = Z_seq.permute(0, 2, 1, 3).reshape(B * N, hat_L, d)
        Z_lstm, _ = self.lstm(Z_seq_resh)
        Z_seq = Z_lstm.reshape(B, N, hat_L, d).permute(0, 2, 1, 3)
        return Z_seq, E_seq

def deep_coral(H_s, H_t):
    n_s, f = H_s.size()
    n_t = H_t.size(0)
    if n_s < 2 or n_t < 2:
        return torch.tensor(0.0, device=H_s.device)
    
    H_s_c = H_s - H_s.mean(dim=0, keepdim=True)
    cov_s = (H_s_c.T @ H_s_c) / (n_s - 1)
    
    H_t_c = H_t - H_t.mean(dim=0, keepdim=True)
    cov_t = (H_t_c.T @ H_t_c) / (n_t - 1)
    
    return torch.norm(cov_s - cov_t, p='fro') ** 2 / (4 * f**2)

class Alignment(nn.Module):
    def __init__(self, lambda_sca=0.01, lambda_sfa=0.01):
        super(Alignment, self).__init__()
        self.lambda_sca = lambda_sca
        self.lambda_sfa = lambda_sfa

    def forward(self, Z_s, E_s, Z_t, E_t):
        B, hat_L, N, d = Z_s.shape
        L_iEndo = 0.0
        
        for T in range(hat_L):
            Z_s_T_sensors = Z_s[:, T].reshape(B * N, d) 
            Z_t_T_sensors = Z_t[:, T].reshape(B * N, d) 
            
            W_T = deep_coral(Z_s_T_sensors, Z_t_T_sensors).detach() 
            
            var_s = torch.var(E_s[:, T], dim=0, unbiased=True)
            var_t = torch.var(E_t[:, T], dim=0, unbiased=True)
            L_iSCA_T = torch.mean((var_s - var_t) ** 2) / 4.0
            
            L_iSFA_T = deep_coral(Z_s_T_sensors, Z_t_T_sensors) 
            
            L_iEndo += W_T * (self.lambda_sca * L_iSCA_T + self.lambda_sfa * L_iSFA_T)
        
        P_s = Z_s.mean(dim=(1, 2))
        P_t = Z_t.mean(dim=(1, 2))
        L_exo = deep_coral(P_s, P_t)
        
        return L_exo + L_iEndo / hat_L

class SimpleGCN(nn.Module):
    def __init__(self, d_model):
        super(SimpleGCN, self).__init__()
        self.W = nn.Linear(d_model, d_model)
        edges = [[0,1],[1,2],[2,3],[0,4],[4,5],[5,6],[0,7],[7,8],[8,9],
                 [9,10],[8,11],[11,12],[12,13],[8,14],[14,15],[15,16]]
        adj = torch.zeros(17, 17)
        for i, j in edges:
            adj[i, j], adj[j, i] = 1, 1
        for i in range(17):
            adj[i, i] = 1
        deg = adj.sum(dim=1, keepdim=True)
        self.register_buffer('adj', adj / deg)

    def forward(self, x):
        x = torch.einsum('ij,bjd->bid', self.adj, x)
        return F.relu(self.W(x))

class PoseDecoder(nn.Module):
    def __init__(self, num_joints=17, d_model=128, num_sensors=342):
        super(PoseDecoder, self).__init__()
        self.mlp = nn.Linear(d_model, d_model)
        self.joint_embedding = nn.Parameter(torch.randn(1, num_joints, d_model))
        
        self.gcn = SimpleGCN(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead=4, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        # 空间三角池化层：提炼 342 个传感器之间的空间差异
        self.spatial_triangulation = nn.Sequential(
            nn.Linear(num_sensors, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        self.root_regressor = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )
        
        self.pose_fc = nn.Linear(d_model, 3)

    def forward(self, Z):
        # ------------------------------------------
        # 分支 A：根节点绝对位置 (保留空间天线差异！)
        # ------------------------------------------
        # Z 的维度: [B, hat_L=27, N=342, d_model=128]
        # 平均掉时间轴 (dim=1)
        Z_time_avg = Z.mean(dim=1) # [B, 342, 128]
        Z_time_avg_T = Z_time_avg.permute(0, 2, 1) # [B, 128, 342]
        
        Z_root = self.spatial_triangulation(Z_time_avg_T).squeeze(-1) 
        root_abs = self.root_regressor(Z_root).unsqueeze(1) # [B, 1, 3]
        
        # ------------------------------------------
        # 分支 B：相对微观动作
        # ------------------------------------------
        Z_agg = Z.mean(dim=1)  
        Z_global = self.mlp(Z_agg.mean(dim=1)) 
        
        Z_joints = Z_global.unsqueeze(1).repeat(1, 17, 1)
        Z_joints = Z_joints + self.joint_embedding
        
        Z_joints_gcn = self.gcn(Z_joints)
        Z_trans = self.transformer(Z_joints_gcn)
        pose_rel = self.pose_fc(Z_trans) 
        
        pose_rel = pose_rel - pose_rel[:, 0:1, :]
        
        # ------------------------------------------
        # 终极融合
        # ------------------------------------------
        poses = pose_rel + root_abs
        return poses

class SEAplusplus(nn.Module):
    def __init__(self, num_sensors=342, d_patch=11, d_model=128, num_branches=3, num_joints=17):
        super(SEAplusplus, self).__init__()
        self.d_patch = d_patch
        self.feature_proj = nn.Linear(d_patch, d_model) 
        self.encoder = GraphEncoder(num_sensors, d_model, num_branches)
        
        self.alignment = Alignment(lambda_sca=0.01, lambda_sfa=0.01)
        self.decoder = PoseDecoder(num_joints, d_model, num_sensors) 

    def forward(self, x_s, x_t=None, train=True):
        B, N, L = x_s.shape
        hat_L = L // self.d_patch
        
        x_s = x_s[:, :, :hat_L * self.d_patch] 
        patches_s = x_s.reshape(B, N, hat_L, self.d_patch).permute(0, 2, 1, 3)
        patches_s = F.relu(self.feature_proj(patches_s))
        
        Z_s, E_s = self.encoder(patches_s)
        poses_s = self.decoder(Z_s)
        
        if not train or x_t is None:
            return poses_s
            
        x_t = x_t[:, :, :hat_L * self.d_patch]
        patches_t = x_t.reshape(B, N, hat_L, self.d_patch).permute(0, 2, 1, 3)
        patches_t = F.relu(self.feature_proj(patches_t))
        
        Z_t, E_t = self.encoder(patches_t)
        align_loss = self.alignment(Z_s, E_s, Z_t, E_t)
        return poses_s, align_loss