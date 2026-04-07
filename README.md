# WiFi-Pose: Cross-Domain Human Pose Estimation via WiFi CSI

基于 WiFi 信道状态信息（CSI）的跨域人体姿态估计，采用对抗域自适应方法实现从源域（已标注环境）到目标域（未标注新环境）的泛化。

---

## 目录

- [项目简介](#项目简介)
- [方法概述](#方法概述)
- [项目结构](#项目结构)
- [环境依赖](#环境依赖)
- [数据集准备](#数据集准备)
- [快速开始](#快速开始)
- [训练](#训练)
- [测试与评估](#测试与评估)
- [可视化分析](#可视化分析)
- [配置说明](#配置说明)
- [评估指标](#评估指标)
- [更新日志](#更新日志)
- [常见问题](#常见问题)

---

## 项目简介

本项目利用 WiFi 设备采集的 CSI 信号，在不依赖摄像头的条件下估计人体 3D 骨架（17 关节）。核心挑战在于**跨域泛化**：不同房间、不同多径环境下 CSI 分布差异显著，导致在源域训练的模型直接迁移到目标域时性能大幅下降。

本方法通过以下机制解决域偏移问题：

- **分类预训练 + 回归微调**（v11）：先用动作分类预训练 Encoder，再冻结 Encoder 训练回归头，最后端到端微调。解决 MSE 回归的均值塌陷问题
- **对抗域自适应**（GRL + 域判别器）：迫使编码器学习域不变的姿态语义特征
- **特征解耦**（DomainDisentangle）：显式分离共享特征与域私有特征，正交损失进一步约束解耦质量
- **跨传感器图注意力**（GAT + CrossSensorAttention）：建模天线间的空间依赖关系
- **时序 Transformer**（含位置编码）：在时间维度上捕捉动作的连续性，每个天线节点独立建模

---

## 方法概述

### 训练流程（v11 三阶段课程学习）

```
Phase 1: 分类预训练 (epoch 0 ~ 14)
  CSI Input → Encoder → ClassificationHead → 27 类动作分类 (CE Loss)
  目的：迫使 Encoder 学会区分不同 CSI 输入（已验证 96% 准确率）

Phase 2: 冻结回归 (epoch 15 ~ 29)
  CSI Input → [Encoder 冻结] → PoseHead → 3D Pose (L1 + Bone Loss)
  目的：PoseHead 必须利用 Encoder 的多样特征，无法退化到均值捷径

Phase 3: 端到端微调 (epoch 30+)
  CSI Input → Encoder(LR×0.1) → PoseHead + ClassificationHead(辅助)
  目的：精细调优，分类辅助损失防止 Encoder 特征退化
```

### 模型架构

```
CSI Input (B, T, N, C)
        │
        ▼
┌───────────────────┐
│  GATLayer          │  ← 空间传感器间图注意力
│  CrossSensorAttn   │  ← 跨传感器自注意力（含残差连接 + LayerNorm）
│  TemporalTF + PE   │  ← 时序 Transformer（每个节点独立，保留节点多样性）
└───────────────────┘
        │  Encoder Output (B, T, N, D)
        ├──► ClassificationHead → 27 类动作 (辅助任务)
        │
        ▼
┌───────────────────┐
│  DomainDisentangle │  → shared_feat  ──► PoseHead → 3D Pose (B, J, 3)
│                    │  → private_feat ──► (仅参与正交损失)
└───────────────────┘
        │
        ├── shared_feat ──► DomainDisc (with GRL) ──► 对抗域分类损失
        └── Orthogonality Loss (shared ⊥ private)
```

> **注意**：域判别器作用于 **shared** 特征（经 GRL 梯度反转），而非 private 特征。
> 这是对抗域自适应的标准做法：GRL 迫使 encoder 让 shared 特征无法被域判别器区分，
> 从而习得域不变表示。

### 损失函数

**Phase 1（分类预训练）：**

```
L_total = CrossEntropy(logits, action_label)
```

**Phase 2（冻结回归）：**

```
L_total = L_pose(L1) + 0.1 × L_vel + 0.1 × L_bone
```

**Phase 3（端到端微调）：**

```
L_total = L_pose(L1) + 0.1 × L_vel + 0.1 × L_bone + 0.5 × L_cls
        [+ 0.01 × L_align + 0.01 × L_domain + 0.01 × L_orth]  (可选 DA 项)
```

| 损失项 | 权重 | 说明 |
|--------|------|------|
| `L_pose` | 1.0 | L1 姿态回归损失（L1 比 MSE 对均值吸引力更弱） |
| `L_vel` | 0.1 | 帧间位移监督（末帧 - 首帧） |
| `L_bone` | 0.1 | 骨骼长度一致性损失 |
| `L_cls` | 0.5 | 动作分类交叉熵（Phase 3 辅助损失，防止 Encoder 退化） |
| `L_align` | 0.01 | 源/目标域 shared 特征均值 MSE 对齐 |
| `L_domain` | 0.01 | 对抗域分类交叉熵（经 GRL，作用于 shared 特征） |
| `L_orth` | 0.01 | shared / private 特征正交约束 |

---

## 项目结构

```
.
├── config/
│   ├── default.yaml            # 默认超参数配置
│   └── exp_cross_domain.yaml   # 跨域实验配置
├── dataset/
│   └── mmfi_dataset.py         # MMFi 数据集加载（CSI 预处理、根节点中心化、预处理缓存）
├── losses/
│   └── losses.py               # 联合损失函数（pose + vel + bone + align + domain + orth）
├── models/
│   ├── attention.py            # CrossSensorAttention（残差连接 + LayerNorm + 注意力权重缓存）
│   ├── disentangle.py          # DomainDisentangle（shared/private 特征分离）
│   ├── domain_disc.py          # 域判别器（内置 GRL，作用于 shared 特征）
│   ├── encoder.py              # 三阶段编码器（GAT + Attn + Transformer）
│   ├── gat.py                  # 图注意力层
│   ├── grad_reverse.py         # 梯度反转层（GRL）
│   ├── model.py                # WiFiPoseModel 主模型
│   ├── pose_head.py            # 姿态回归头（v5 修复：max-pool + motion 输入）
│   └── transformer.py          # TemporalTransformer（含正弦位置编码，节点独立建模）
├── utils/
│   ├── checkpoint.py           # checkpoint 保存与加载（strict=False，详细加载报告）
│   └── config.py               # YAML + 命令行参数合并
├── visualization/
│   ├── attention_vis.py        # 传感器注意力热力图与统计
│   ├── experiment_report.py    # 多实验对比报告
│   ├── joint_error_analysis.py # 逐关节误差 / 雷达图 / 骨架热力图
│   ├── plot_loss.py            # 简易损失曲线（服务器友好，savefig 接口）
│   ├── plot_training_curves.py # 训练曲线可视化
│   ├── pose_vis.py             # 3D 姿态对比 / 序列 / 误差向量场
│   └── tsne_analysis.py        # t-SNE 域对齐分析
├── train_v11.py                # 🔥 推荐训练脚本：分类预训练 + 回归微调
├── train.py                    # 原始训练主脚本（含 DA，v8 课程学习版）
├── train_source_only.py        # 源域纯监督训练（基线验证）
├── diagnose_data.py            # 数据诊断脚本（CSI 信号质量、动作分类、帧对齐）
├── diagnose_collapse.py        # 预测塌陷诊断脚本
├── test_cross_domain.py        # 跨域批量测试
├── test_single.py              # 单样本测试 + 可视化
├── run_all_vis.py              # 一键运行所有可视化分析
└── utils_metrics.py            # MPJPE / PA-MPJPE / PCK 评估指标
```

---

## 环境依赖

**Python 版本：** 3.8+

```bash
pip install torch torchvision          # PyTorch >= 1.12（推荐 2.x）
pip install numpy scipy matplotlib
pip install scikit-learn               # t-SNE / silhouette score
pip install pyyaml tqdm                # tqdm 用于缓存构建进度显示
```

**GPU 建议：** 训练需要至少 16GB 显存（batch_size=64，dim=256）。如显存不足可降低 batch_size 或 dim。

验证安装：

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

---

## 数据集准备

本项目使用 [MMFi 数据集](https://github.com/ybhbingo/MMFi_dataset)。

**目录结构要求：**

```
/path/to/MMFi/
├── E01/
│   ├── S01/
│   │   ├── A01/
│   │   │   ├── wifi-csi/
│   │   │   │   ├── frame_000001.mat
│   │   │   │   └── ...
│   │   │   └── ground_truth.npy   # shape: (T, 17, 3)
│   │   └── A02/
│   │       └── ...
│   └── S02/
│       └── ...
├── E02/
├── E03/
└── E04/
```

下载完成后，修改 `config/default.yaml` 中的 `data.root` 为实际路径：

```yaml
data:
  root: /path/to/MMFi
```

**强烈建议开启预处理缓存**（可显著提升 GPU 利用率，详见[配置说明](#配置说明)）：

```yaml
data:
  root: /path/to/MMFi
  cache_dir: /path/to/MMFi_cache   # 缓存目录，首次运行自动构建
```

---

## 快速开始

### 数据诊断（推荐首次运行）

在训练之前，建议先运行数据诊断脚本确认 CSI 数据质量：

```bash
python diagnose_data.py
```

该脚本检查 GT 姿态统计、CSI 信号质量、动作分类准确率（应 >90%）、帧对齐等。

### 训练（推荐：v11 分类预训练）

```bash
python train_v11.py --epochs 60
```

### 跨域测试

```bash
python test_cross_domain.py --work ./outputs/<exp_name>/best.pth
```

### 一键可视化

```bash
python run_all_vis.py \
    --checkpoint ./outputs/<exp_name>/best.pth \
    --data_root /path/to/MMFi \
    --save_dir ./vis_output
```

---

## 训练

### 推荐方式：v11 分类预训练（解决均值塌陷）

```bash
# 默认配置：Phase1=15ep + Phase2=15ep + Phase3=剩余
python train_v11.py --epochs 60

# 自定义各阶段长度
python train_v11.py --epochs 80 --phase1_epochs 20 --phase2_epochs 20

# 命令行参数覆盖
python train_v11.py --batch_size 32 --lr 3e-4 --epochs 80
```

**v11 三阶段详解：**

| 阶段 | Epoch | Encoder | PoseHead | 损失 | 目的 |
|------|-------|---------|----------|------|------|
| Phase 1 | 0-14 | 训练 | — | CE 分类 | Encoder 学会区分不同 CSI |
| Phase 2 | 15-29 | **冻结** | 训练 | L1+bone | PoseHead 利用多样特征 |
| Phase 3 | 30+ | 微调(LR×0.1) | 训练 | L1+bone+0.5×CE | 端到端精调 |

**训练日志示例：**

```
# Phase 1 (分类)
[E005][0100/9348] [Phase1-CLS] loss=0.3142  cls=0.3142
  [SRC-CLS] Action accuracy: 94.2%
  [TGT-CLS] Action accuracy: 88.7%

# Phase 2 (冻结回归)
[E015][0100/9348] [Phase2-REG(frozen)] loss=0.0423  pose=0.0356  vel=0.0412  bone=0.0158
  [Collapse] pred_std=35.2mm  gt_std=29.7mm  ratio=1.184  ✓正常

# Phase 3 (微调)
[E035][0100/9348] [Phase3-FINETUNE] loss=0.0387  pose=0.0312  vel=0.0389  bone=0.0135  cls=0.2841
```

> **Phase 1 结束时分类准确率应 >90%。如果远低于此，说明数据或预处理有问题，需先运行 `diagnose_data.py`。**

### 备选方式：原始 DA 训练

```bash
python train.py \
    --config config/default.yaml \
    --source E01 E02 E03 \
    --target E04
```

### 从 checkpoint 恢复

```bash
python train_v11.py --resume ./outputs/20240101_120000/latest.pth
```

**训练输出：**

```
outputs/
└── v11_clspretrain_20240101_120000/
    ├── config.yaml     # 本次实验完整配置（自动保存）
    ├── latest.pth      # 最新 epoch 权重
    └── best.pth        # 最低源域 MPJPE 权重
```

Checkpoint 中额外保存以下字段，供测试脚本自动恢复模型结构：

| 字段 | 说明 |
|------|------|
| `in_dim` | 编码器输入维度（P×C，由数据集自动探测） |
| `pose_head_old` | PoseHead 结构标志：`False` = 新版（max-pool + motion），`True` = 旧版（concat x0+xT） |
| `phase` | 训练阶段（1/2/3），仅 v11 |

---

## 测试与评估

### 跨域批量测试

```bash
python test_cross_domain.py --work ./outputs/<exp_name>/best.pth

# 指定目标域
python test_cross_domain.py --work ./outputs/<exp_name>/best.pth --target E04
```

测试脚本会自动从 checkpoint 中读取 `in_dim` 和 `pose_head_old`，无需手动指定模型结构。对于不含这些字段的旧版 checkpoint，脚本会通过权重形状自动反推，保持向后兼容。

输出示例：

```
===== RESULTS =====
MPJPE:    0.0720
PA-MPJPE: 0.0580
PCK@0.05: 0.5430
```

### 单样本测试（含 3D 可视化）

```bash
python test_single.py \
    --work ./outputs/<exp_name>/best.pth \
    --env E04 \
    --subject S35 \
    --action A01
```

生成 `single_pose.png`：GT（绿色）与预测（红色）的 3D 骨架对比图，预测关节按误差大小着色。

### 预测塌陷诊断

```bash
python diagnose_collapse.py --work ./outputs/<exp_name>/best.pth
```

输出预测方差、样本对距离、特征余弦相似度等指标，判断模型是否塌陷到均值。

---

## 可视化分析

### 一键运行全部分析

```bash
python run_all_vis.py \
    --checkpoint ./outputs/<exp_name>/best.pth \
    --data_root /path/to/MMFi \
    --save_dir ./vis_output \
    --source E01 E02 E03 \
    --target E04
```

生成以下图表：

| 文件 | 内容 |
|------|------|
| `single_pose_sample.png` | 单帧 GT vs 预测 3D 骨架对比，关节用误差着色 |
| `pose_sequence.png` | 连续 30 帧的姿态序列对比（均匀抽取 6 帧） |
| `error_vector_field.png` | 正/侧视图误差向量场，显示各关节偏移方向 |
| `joint_error_heatmap.png` | 17 关节 MPJPE 排序条形图 |
| `body_part_radar.png` | 六大身体部位误差雷达图 |
| `skeleton_error_heatmap.png` | 骨架图上的误差热力图（颜色深 = 误差大） |
| `tsne_domain_alignment.png` | t-SNE 域对齐图（含 silhouette score） |
| `sensor_attention_mean.png` | 所有样本平均传感器注意力矩阵 |
| `attention_statistics.png` | 自注意力强度分布 + 各节点注意力熵 |

### 单独运行各模块

**训练曲线对比：**

```bash
python -m visualization.plot_training_curves \
    --log_dir ./outputs/run1 ./outputs/run2 \
    --labels baseline improved \
    --save training_compare.png
```

**多实验定量对比报告：**

```bash
python -m visualization.experiment_report \
    --checkpoints ./outputs/run1/best.pth ./outputs/run2/best.pth \
    --labels "w/o pretrain" "v11 pretrain" \
    --data_root /path/to/MMFi \
    --target E04 \
    --save_dir ./vis_output
```

---

## 配置说明

`config/default.yaml` 完整字段说明：

```yaml
data:
  root: /path/to/MMFi        # 数据集根目录
  seq_len: 20                # 输入时间窗口长度（帧数）
  cache_dir: /path/to/cache  # 预处理缓存目录（可选，强烈建议开启）
                             # 首次运行自动构建，后续直接读 .npy，GPU 利用率从 <5% 提升至 70%+

train:
  batch_size: 64             # 每批样本数（16GB 显存建议 64，dim=256）
  epochs: 50                 # 训练总轮数
  lr: 0.0001                 # Adam 初始学习率（Cosine 衰减至 1e-6）

model:
  dim: 256                   # 编码器隐层维度（GAT / Transformer / PoseHead 统一）
  num_joints: 17             # 骨架关节数（H36M 标准）

domain:
  source: [E01, E02, E03]   # 源域环境列表（有标注）
  target: [E04]             # 目标域环境列表（无标注）

log:
  save_dir: ./outputs        # 实验输出根目录
  print_freq: 20             # 每隔多少 iter 打印一次日志
```

> `in_dim` 无需在配置文件中指定，训练脚本会在启动时自动从数据集第一个样本探测 `CSI.shape[-1]`，并将结果写入 checkpoint。

**v11 额外命令行参数：**

| 参数 | 默认值 | 说明 |
|------|-------|------|
| `--phase1_epochs` | 15 | Phase 1 分类预训练持续的 epoch 数 |
| `--phase2_epochs` | 15 | Phase 2 冻结回归持续的 epoch 数 |

**命令行参数优先级：** 命令行 > `--exp` 实验配置 > `--config` 基础配置

**参数说明 — `--resume` 与 `--work` 的区别：**

| 参数 | 用于脚本 | 作用 |
|------|---------|------|
| `--resume` | `train.py` / `train_v11.py` | 从中断点恢复训练，同时加载模型权重、优化器状态和 epoch 编号 |
| `--work` | `test_single.py` / `test_cross_domain.py` | 仅加载模型权重用于推理，不涉及训练状态 |

---

## 评估指标

| 指标 | 说明 | 越小越好 |
|------|------|---------|
| **MPJPE** | 均关节位置误差，预测与真值的平均欧氏距离（mm）。本项目在根节点中心化后的相对坐标空间计算，已消除全局位置偏移影响 | ✓ |
| **PA-MPJPE** | Procrustes 对齐后的 MPJPE，额外消除全局旋转/缩放影响，衡量纯姿态形状精度。数学上必须 ≤ MPJPE | ✓ |
| **PCK@0.05** | 关节误差 < 0.05m 的比例，衡量粗粒度准确率 | 越大越好 |

---

## 更新日志

**v11（当前版本）**

> 彻底解决困扰 v1-v10 所有版本的均值塌陷问题。

**核心变化：分类预训练 + 三阶段课程学习**

- **根因定位**：通过数据诊断发现 CSI 动作分类准确率高达 96%（27 类），证明 CSI 信号丰富、数据无问题。均值塌陷的根本原因是 MSE/L1 回归存在"均值捷径"（输出均值姿态即可获得不错的 loss），而交叉熵分类不存在此问题
- **新增 `train_v11.py`**：三阶段训练脚本
  - Phase 1：动作分类预训练 Encoder（CE loss，必然成功）
  - Phase 2：冻结 Encoder，单独训练 PoseHead（L1 + bone loss）
  - Phase 3：解冻全部，Encoder 用 0.1× 小 LR 微调，保留分类辅助损失
- **新增 `diagnose_data.py`**：数据诊断脚本，检查 GT 统计、CSI 信号质量、动作分类、帧对齐、预处理影响
- **新增 `diagnose_collapse.py`**：预测塌陷诊断脚本
- **L1 替代 MSE**：L1 对均值的吸引力弱于 MSE，配合分类预训练进一步减弱塌陷倾向

**为什么之前的方法（v6-v10）没有解决塌陷：**

| 版本 | 尝试的方法 | 失败原因 |
|------|-----------|---------|
| v6 | Diversity loss（批内预测多样性） | 高维空间中 L2 margin 设置不当，损失从未激活 |
| v7 | VICReg 方差正则 | 源域特征确实多样了，但目标域仍塌陷；batch 内 std>1 太容易满足 |
| v8 | IO 一致性损失 | 需要先解决源域塌陷问题；课程学习的 DA warmup 不够 |
| v9 | 特征距离匹配 | Encoder 特征多样但 PoseHead 学会零权重忽略输入 |
| v10 | 预测空间距离匹配 + Flat MLP | torch.cdist 在零距离处梯度为 0，无法逃离塌陷点；Flat MLP 也塌陷证明是训练策略而非架构问题 |

---

**v5**

> 修复 PoseHead 均值塌陷的首次尝试。

- `models/pose_head.py`：PoseHead 输入改为 `concat(x_max, motion)`，替代 `concat(x0, xT)`。`x_max` 为全序列时间 max-pooling，`motion` 为首末帧差分。保留 `old_style` 参数兼容旧 checkpoint
- `models/attention.py`：CrossSensorAttention 新增残差连接 + LayerNorm，防止注意力趋于均匀时所有节点被平均为相同向量
- `losses/losses.py`：新增 `compute_diversity_loss`（批内预测多样性约束）
- `models/model.py`：forward 新增目标域预测 `pose_t`，供 diversity loss 约束
- `train.py`：GRL warmup 机制（前 N 个 epoch alpha=0），防止对抗梯度过早干扰 Encoder 学习

---

**v4**

> 修复了 v3 中三处影响评估正确性和预测质量的 bug。

**Critical 修复：**

- `utils_metrics.py`：修复 `compute_similarity_transform` 中三处同时错误的 Procrustes 公式，导致 PA-MPJPE 数学上大于 MPJPE 的异常结果。具体为：
  1. `det` 的参数从 `det(Vh.T @ U.T)` 修正为 `det(U @ Vh)`
  2. 旋转矩阵从 `R = Vh.T @ sign_fix @ U.T` 修正为 `R = U @ sign_fix @ Vh`
  3. 返回值从 `@ R.T` 修正为 `@ R`（与 `H = X0.T @ Y0` 的约定一致）

- `models/pose_head.py`：解耦姿态预测与速度预测。原版将 `concat([x0, xT])` 输入姿态头，导致末帧特征将四肢"拉向"末帧位置。修复后姿态头仅使用 `x0`，速度头仅使用 `(xT - x0)`

- `models/model.py`：`in_dim` 不再硬编码为 `40`，改为外部传入参数

- `test_cross_domain.py` / `test_single.py`：新增 `detect_model_dims()` 函数，自动反推模型结构参数，完全向后兼容

---

**v3**

> 修复了 v2 中存在的若干 critical bug 及设计问题。

**Critical 修复：**

- `utils/checkpoint.py`：`load_state_dict` 增加 `strict=False`
- `visualization/*.py`：修复 DataLoader 三元组解包错误、`gat.fc` 不存在等问题

**Design 修复：**

- `models/model.py`：域判别器改为作用于 **shared** 特征（原版作用于 private 特征，缺乏对抗意义）
- `models/transformer.py`：`TemporalTransformer` 改为对每个节点独立建模（`(B*N, T, D)`）
- `losses/losses.py`：帧间位移监督权重从 0.5 降至 0.1

**Minor 修复：**

- 可视化脚本 `--dim` 默认值统一为 256
- `dataset/mmfi_dataset.py`：保留天线维度 `(N, P, 4)` 供模型学习
- `visualization/plot_loss.py`：`plt.show()` 改为 `plt.savefig()`

---

**v2**
- `mmfi_dataset.py`：新增根节点中心化，解决 MPJPE 与 PA-MPJPE 差距过大问题
- `mmfi_dataset.py`：新增预处理缓存（`cache_dir`），GPU 利用率从 <5% 提升至 70%+
- `losses.py`：新增骨骼长度一致性损失（`L_bone`）
- `test_single.py` / `test_cross_domain.py`：测试脚本统一改为 `--work` 参数

**v1**
- 初始版本：GAT + CrossSensorAttention + TemporalTransformer 编码器
- 对抗域自适应（GRL + DomainDisentangle + 正交损失）
- 修复原始代码中 domain loss 空操作、梯度反转层缺失、维度硬编码等 4 个 critical bug

---

## 常见问题

**Q: 模型预测所有样本几乎相同的骨架（均值塌陷），怎么办？**

这是 MSE/L1 回归的经典问题。使用 `train_v11.py` 的分类预训练方案可以解决：
```bash
python train_v11.py --epochs 60
```
原理：先用动作分类（27 类）预训练 Encoder，迫使其学会区分不同 CSI 输入，再冻结 Encoder 训练回归头。详见[训练](#训练)。

可用以下脚本诊断塌陷状态：
```bash
python diagnose_collapse.py --work ./outputs/<exp_name>/best.pth
```

**Q: 如何判断 CSI 数据中是否有姿态信号？**

运行数据诊断脚本：
```bash
python diagnose_data.py
```
关键指标是 Test 3 的动作分类准确率。如果 >20%（随机基线 3.7%），说明 CSI 包含运动信息。我们的实验中达到了 96%。

**Q: MPJPE 很高但 PA-MPJPE 正常，是什么原因？**

这是**全局位置偏移**问题。PA-MPJPE 在计算前会对齐全局位置，如果两者差距很大（如 MPJPE=344mm，PA-MPJPE=108mm），说明模型学到了骨架形状，但预测的绝对位置偏移很大。本项目已通过**根节点中心化**解决此问题——训练和测试均在相对坐标空间进行，MPJPE 应接近 PA-MPJPE 量级。

**Q: PA-MPJPE 大于 MPJPE，这正常吗？**

不正常。PA-MPJPE 经过 Procrustes 对齐，数学上必须 ≤ MPJPE。出现反转说明 `utils_metrics.py` 中使用了旧版错误公式。请确认已替换为 v4 版本。

**Q: Phase 1 分类准确率很低（<50%），怎么办？**

说明 Encoder 无法从 CSI 中提取有意义的特征。排查步骤：
1. 运行 `diagnose_data.py`，检查 Test 3 的 Flat MLP 分类准确率
2. 如果 Flat MLP 也 <50%，问题在数据预处理或帧对齐
3. 如果 Flat MLP >80% 但 Encoder <50%，问题在 Encoder 架构

**Q: Phase 2 开始后仍然塌陷（pred_std 很低）？**

确认 Phase 1 的分类准确率 >90%。如果分类准确率高但回归仍塌陷：
- 增加 Phase 2 的 epoch 数（`--phase2_epochs 25`）
- 尝试减小 PoseHead 的层数或维度
- 检查 Encoder 是否确实被冻结（日志中应显示 `frozen`）

**Q: 加载 checkpoint 时报 `size mismatch for head.mlp.0.weight`，怎么处理？**

这是新旧 `PoseHead` 结构不兼容导致的。v4+ 的测试脚本已内置自动检测逻辑，会通过 checkpoint 中权重的形状反推正确结构并自动适配。

**Q: GPU 利用率很低（< 10%）？**

开启预处理缓存：

```yaml
# config/default.yaml
data:
  cache_dir: /path/to/MMFi_cache
```

首次运行会自动构建缓存（约需 10-30 分钟），之后 GPU 利用率可从 <5% 提升至 70%+。

**Q: 训练时出现 NaN loss？**

代码已内置 NaN batch 跳过机制，并在数据加载时做了 `nan_to_num` 清理。如果频繁出现：
- 适当减小学习率（`--lr 5e-5`）
- 检查 `.mat` 文件是否有损坏帧

**Q: 显存不足（OOM）？**

- 减小 `batch_size`（`--batch_size 32`）
- 减小 `model.dim`（`dim: 128`）
- 在 `mmfi_dataset.py` 中加大子载波降采样倍率（`::4` 替换 `::2`）

**Q: 旧版 checkpoint 能用于新代码吗？**

可以。测试脚本对旧版 checkpoint 完全向后兼容：
- 若 checkpoint 不含 `in_dim` 字段，脚本自动通过 `encoder.gat.proj.weight` 的形状反推
- 若 checkpoint 不含 `pose_head_old` 字段，脚本自动通过 `head.mlp.0.weight` 的输入维度判断新旧结构

**Q: domain loss 一直很高（远大于 1.386）？**

说明域判别器仍能轻松区分源域与目标域，域对齐尚未成功。可以：
- 增大训练 epoch（GRL 的 alpha 随 epoch 线性增大）
- 确保域判别器正确接在 **shared** 特征上（v3 已修复）
- 先使用 v11 方案确保基础回归能力，再叠加 DA 组件

**Q: 可视化脚本运行时报 "too many values to unpack"？**

`MMFiDataset.__getitem__` 返回三元组 `(csi, pose, root_offset)`。v3 已全部修复。

**Q: t-SNE 运行很慢？**

```bash
python run_all_vis.py ... --n_tsne_samples 500
```

**Q: 如何添加新的源域/目标域组合？**

```bash
python train_v11.py --source E01 E02 --target E03 --epochs 60
python test_cross_domain.py --work best.pth --target E03
```