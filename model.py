import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
import math

class MSGC(nn.Module):  # 多分支自注意力图构建模块
    def __init__(self, d_model, num_branches=3):
        super(MSGC, self).__init__()
        self.num_branches = num_branches
        self.queries = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(num_branches)])
        self.keys = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(num_branches)])

    def forward(self, Z_T):  # Z_T: [N, d]
        E_T_branches = []
        for i in range(self.num_branches):
            Q = self.queries[i](Z_T)  # [N, d]
            K = self.keys[i](Z_T)  # [N, d]
            attn = torch.matmul(Q, K.T) / math.sqrt(Z_T.size(-1))  # [N, N]
            E_i = F.softmax(attn, dim=-1)
            E_T_branches.append(E_i)
        E_T = torch.mean(torch.stack(E_T_branches), dim=0)  # 平均分支
        return E_T

class MPNN(nn.Module):  # 消息传递神经网络
    def __init__(self, d_model):
        super(MPNN, self).__init__()
        self.mlp = nn.Linear(d_model, d_model)

    def forward(self, Z_T, E_T):  # Z_T: [N, d], E_T: [N, N]
        h = torch.matmul(E_T, Z_T)  # 消息传递 [N, d]
        Z_T_updated = F.relu(self.mlp(h))
        return Z_T_updated

class GraphEncoder(nn.Module):  # 图基编码器
    def __init__(self, num_sensors, d_model, num_branches=3, lstm_layers=2):
        super(GraphEncoder, self).__init__()
        self.msgc = MSGC(d_model, num_branches)
        self.mpnn = MPNN(d_model)
        self.lstm = nn.LSTM(d_model, d_model, lstm_layers, batch_first=True, bidirectional=False)

    def forward(self, patches):  # patches: [B, hat_L, N, d]
        B, hat_L, N, d = patches.shape
        Z_seq = []
        E_seq = []
        for T in range(hat_L):
            Z_T = patches[:, T]  # [B, N, d]
            E_T = self.msgc(Z_T.mean(0))  # 计算相关性
            Z_T = self.mpnn(Z_T, E_T.unsqueeze(0).repeat(B, 1, 1))
            Z_seq.append(Z_T)
            E_seq.append(E_T)
        Z_seq = torch.stack(Z_seq, dim=1)  # [B, hat_L, N, d]
        # LSTM处理每个传感器的时间序列
        Z_seq_resh = Z_seq.permute(0, 2, 1, 3).reshape(B*N, hat_L, d)  # [B*N, hat_L, d]
        Z_lstm, _ = self.lstm(Z_seq_resh)
        Z_seq = Z_lstm.reshape(B, N, hat_L, d).permute(0, 2, 1, 3)  # [B, hat_L, N, d]
        E_seq = torch.stack(E_seq, dim=1).unsqueeze(0).repeat(B, 1, 1, 1)  # [B, hat_L, N, N]
        return Z_seq, E_seq

def deep_coral(H_s, H_t):  # Deep Coral损失，用于高阶统计对齐
    n_s = H_s.size(0)
    n_t = H_t.size(0)
    f = H_s.size(1)
    cov_s = torch.cov(H_s.T)
    cov_t = torch.cov(H_t.T)
    return torch.norm(cov_s - cov_t, p='fro') ** 2 / (4 * f**2)

class Alignment(nn.Module):  # 对齐模块
    def __init__(self, lambda_sca=0.1, lambda_sfa=0.1):
        super(Alignment, self).__init__()
        self.lambda_sca = lambda_sca
        self.lambda_sfa = lambda_sfa

    def forward(self, Z_s, E_s, Z_t, E_t):  # All [B, hat_L, N, d] or [B, hat_L, N, N]
        B, hat_L, N, d = Z_s.shape
        L_iEndo = 0
        W_T_list = []
        for T in range(hat_L):
            # 计算W_T: 第T图的域间差异
            Z_s_T_flat = Z_s[:, T].reshape(B, N*d)
            Z_t_T_flat = Z_t[:, T].reshape(B, N*d)
            W_T = deep_coral(Z_s_T_flat, Z_t_T_flat)
            W_T_list.append(W_T)
            
            # iSCA,T: 传感器相关性对齐
            L_iSCA_T = 0
            for m in range(N):
                for n in range(N):
                    E_s_mn_T = E_s[:, T, m, n].unsqueeze(1)  # [B, 1]
                    E_t_mn_T = E_t[:, T, m, n].unsqueeze(1)
                    L_iSCA_T += deep_coral(E_s_mn_T, E_t_mn_T)
            L_iSCA_T /= (N * N)
            
            # iSFA,T: 传感器特征对齐（对比式）
            L_iSFA_T = 0
            for n in range(N):
                Z_s_n_T = Z_s[:, T, n]  # [B, d]
                pos = deep_coral(Z_s_n_T, Z_t[:, T, n])
                neg_sum = sum(deep_coral(Z_s_n_T, Z_t[:, T, j]) for j in range(N) if j != n)
                L_iSFA_T += -torch.log(torch.exp(pos) / (torch.exp(pos) + neg_sum + 1e-8)) / N
            L_iEndo += W_T * (self.lambda_sca * L_iSCA_T + self.lambda_sfa * L_iSFA_T)
        
        # Exo: 全局特征对齐
        P_s = Z_s.reshape(B, hat_L * N * d)  # 堆叠
        P_t = Z_t.reshape(B, hat_L * N * d)
        L_exo = deep_coral(P_s, P_t)
        
        return L_exo + L_iEndo / hat_L  # 平均T

class PoseDecoder(nn.Module):  # 姿态解码器
    def __init__(self, num_joints=17, d_model=128):
        super(PoseDecoder, self).__init__()
        self.mlp = nn.Linear(d_model, d_model)
        self.gcn = GCNConv(d_model, d_model)  # GCN层
        self.transformer = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model, nhead=4), num_layers=2)
        self.fc = nn.Linear(d_model, 3)  # 输出3D坐标
        # 人体骨骼边（17关节，COCO风格示例）
        self.edge_index = torch.tensor([[0,1],[1,2],[2,3],[0,4],[4,5],[5,6],[0,7],[7,8],[8,9],[9,10],[8,11],[11,12],[12,13],[8,14],[14,15],[15,16]], dtype=torch.long).t().contiguous()

    def forward(self, Z):  # Z: [B, hat_L, N, d] → 聚合
        Z_agg = Z.mean(1)  # 平均hat_L [B, N, d]
        Z_agg = self.mlp(Z_agg.mean(1))  # 全局池化 [B, d]
        Z_joints = Z_agg.unsqueeze(1).repeat(1, 17, 1)  # [B, 17, d]
        # GCN
        Z_joints_gcn = self.gcn(Z_joints.view(-1, d_model), self.edge_index.repeat(1, Z_joints.size(0)))
        Z_joints_gcn = Z_joints_gcn.view(B, 17, d_model)
        # Transformer
        Z_trans = self.transformer(Z_joints_gcn.permute(1, 0, 2))  # [17, B, d]
        Z_trans = Z_trans.permute(1, 0, 2)  # [B, 17, d]
        poses = self.fc(Z_trans)  # [B, 17, 3]
        return poses

class SEAplusplus(nn.Module):  # SEA++模型
    def __init__(self, num_sensors=342, d_model=128, num_branches=3, num_joints=17, lambda_sca=0.1, lambda_sfa=0.1):
        super(SEAplusplus, self).__init__()
        self.embed = nn.Linear(1, d_model)  # 嵌入层
        self.encoder = GraphEncoder(num_sensors, d_model, num_branches)
        self.alignment = Alignment(lambda_sca, lambda_sfa)
        self.decoder = PoseDecoder(num_joints, d_model)

    def forward(self, x_s, x_t=None, train=True):  # x: [B, N, L]
        B, N, L = x_s.shape
        d = 8  # patch大小
        hat_L = L // d
        patches_s = x_s.reshape(B, N, hat_L, d).permute(0, 2, 1, 3)  # [B, hat_L, N, d]
        Z_s, E_s = self.encoder(patches_s)
        poses_s = self.decoder(Z_s)
        if not train or x_t is None:
            return poses_s
        patches_t = x_t.reshape(B, N, hat_L, d).permute(0, 2, 1, 3)
        Z_t, E_t = self.encoder(patches_t)
        align_loss = self.alignment(Z_s, E_s, Z_t, E_t)
        return poses_s, align_loss