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

- **对抗域自适应**（GRL + 域判别器）：迫使编码器学习域不变的姿态语义特征
- **特征解耦**（DomainDisentangle）：显式分离共享特征与域私有特征，正交损失进一步约束解耦质量
- **跨传感器图注意力**（GAT + CrossSensorAttention）：建模天线间的空间依赖关系
- **时序 Transformer**（含位置编码）：在时间维度上捕捉动作的连续性

---

## 方法概述

```
CSI Input (B, T, N, C)
        │
        ▼
┌───────────────────┐
│  GATLayer          │  ← 空间传感器图注意力
│  CrossSensorAttn   │  ← 跨传感器自注意力
│  TemporalTF + PE   │  ← 时序 Transformer（仅在 T 维度）
└───────────────────┘
        │  Encoder Output (B, T, N, D)
        ▼
┌───────────────────┐
│  DomainDisentangle │  → shared_feat  ──► PoseHead → 3D Pose (B, J, 3)
│                    │  → private_feat ──► DomainDisc (with GRL)
└───────────────────┘
        │
  Orthogonality Loss (shared ⊥ private)
```

**损失函数：**

```
L_total = L_pose + 0.1 × L_bone + 0.01 × L_align + 0.01 × L_domain + 0.01 × L_orth
```

| 损失项 | 权重 | 说明 |
|--------|------|------|
| `L_pose` | 1.0 | MSE 姿态回归损失，对根节点中心化后的相对坐标监督 |
| `L_bone` | 0.1 | 骨骼长度一致性损失，约束预测骨架各骨骼长度与 GT 一致 |
| `L_align` | 0.01 | 源/目标域共享特征均值 MSE 对齐 |
| `L_domain` | 0.01 | 对抗域分类交叉熵（源域=0，目标域=1，经过 GRL） |
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
│   └── losses.py               # 联合损失函数（pose + bone + align + domain + orth）
├── models/
│   ├── attention.py            # CrossSensorAttention（注意力权重缓存）
│   ├── disentangle.py          # DomainDisentangle（shared/private 特征分离）
│   ├── domain_disc.py          # 域判别器（内置 GRL）
│   ├── encoder.py              # 三阶段编码器（GAT + Attn + Transformer）
│   ├── gat.py                  # 图注意力层
│   ├── grad_reverse.py         # 梯度反转层（GRL）
│   ├── model.py                # WiFiPoseModel 主模型
│   ├── pose_head.py            # 姿态回归头
│   └── transformer.py          # TemporalTransformer（含正弦位置编码）
├── utils/
│   ├── checkpoint.py           # checkpoint 保存与加载（支持新旧结构部分复用）
│   └── config.py               # YAML + 命令行参数合并
├── visualization/
│   ├── attention_vis.py        # 传感器注意力热力图与统计
│   ├── experiment_report.py    # 多实验对比报告
│   ├── joint_error_analysis.py # 逐关节误差 / 雷达图 / 骨架热力图
│   ├── plot_training_curves.py # 训练曲线可视化
│   ├── pose_vis.py             # 3D 姿态对比 / 序列 / 误差向量场
│   └── tsne_analysis.py        # t-SNE 域对齐分析
├── train.py                    # 训练主脚本
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

**GPU 建议：** 训练需要至少 12GB 显存（batch_size=32，dim=64）。batch_size=64 需要 16GB 以上（如 RTX 4080）。

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

### 训练（默认配置）

```bash
python train.py
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

### 基础训练

```bash
python train.py \
    --config config/default.yaml \
    --source E01 E02 E03 \
    --target E04
```

### 命令行参数覆盖

```bash
python train.py \
    --batch_size 16 \
    --lr 5e-5 \
    --epochs 100
```

### 从 checkpoint 恢复

```bash
python train.py --resume ./outputs/20240101_120000/latest.pth
```

### 使用实验配置文件

```bash
python train.py \
    --config config/default.yaml \
    --exp config/exp_cross_domain.yaml
```

**训练输出：**

```
outputs/
└── 20240101_120000/
    ├── config.yaml     # 本次实验完整配置（自动保存）
    ├── latest.pth      # 最新 epoch 权重
    ├── best.pth        # 最低验证损失权重
    └── epoch_N.pth     # 各 epoch 权重
```

**训练日志示例：**

```
[Epoch 010][0020/3505] loss=0.0312  pose=0.0198  bone=0.0087  align=0.0003  domain=1.3821  orth=0.0000  alpha=0.200
 Epoch 010 done | avg=0.0301 | pose=0.0192 | bone=0.0083 | align=0.0003 | domain=1.3756 | orth=0.0000
  → new best: 0.0301
```

> `domain ≈ 1.386` 是正常现象，对应二分类随机猜测的交叉熵（`-log(0.5) × 2`），说明判别器无法区分源域与目标域，**域对齐成功**。

---

## 测试与评估

### 跨域批量测试

```bash
python test_cross_domain.py --work ./outputs/<exp_name>/best.pth

# 指定目标域
python test_cross_domain.py --work ./outputs/<exp_name>/best.pth --target E04
```

输出示例（根节点中心化后的相对坐标评估）：

```
===== RESULTS =====
MPJPE:    0.1200
PA-MPJPE: 0.0950
PCK@0.05: 0.4320
```

### 单样本测试（含 3D 可视化）

```bash
python test_single.py \
    --work ./outputs/<exp_name>/best.pth \
    --env E04 \
    --subject S35 \
    --action A01
```

生成 `single_pose.png`：GT（绿色）与预测（红色）的 3D 骨架对比图。

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
    --labels "w/o GRL" "full model" \
    --data_root /path/to/MMFi \
    --target E04 \
    --save_dir ./vis_output
```

生成柱状对比图和 `report.json`，方便写论文表格。

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
  batch_size: 32             # 每批样本数（RTX 4080 16GB 建议 32~64）
  epochs: 50                 # 训练总轮数
  lr: 0.0001                 # Adam 初始学习率（Cosine 衰减至 1e-6）

model:
  dim: 64                    # 编码器隐层维度（GAT / Transformer / PoseHead 统一）
  num_joints: 17             # 骨架关节数（H36M 标准）

domain:
  source: [E01, E02, E03]   # 源域环境列表（有标注）
  target: [E04]             # 目标域环境列表（无标注）

log:
  save_dir: ./outputs        # 实验输出根目录
  print_freq: 20             # 每隔多少 iter 打印一次日志
```

**命令行参数优先级：** 命令行 > `--exp` 实验配置 > `--config` 基础配置

**参数说明 — `--resume` 与 `--work` 的区别：**

| 参数 | 用于脚本 | 作用 |
|------|---------|------|
| `--resume` | `train.py` | 从中断点恢复训练，同时加载模型权重、优化器状态和 epoch 编号 |
| `--work` | `test_single.py` / `test_cross_domain.py` | 仅加载模型权重用于推理，不涉及训练状态 |

---

## 评估指标

| 指标 | 说明 | 越小越好 |
|------|------|---------|
| **MPJPE** | 均关节位置误差，预测与真值的平均欧氏距离（mm）。本项目在根节点中心化后的相对坐标空间计算，已消除全局位置偏移影响 | ✓ |
| **PA-MPJPE** | Procrustes 对齐后的 MPJPE，额外消除全局旋转/缩放影响，衡量纯姿态形状精度 | ✓ |
| **PCK@0.05** | 关节误差 < 0.05m 的比例，衡量粗粒度准确率 | 越大越好 |

---

## 更新日志

**v2（当前版本）**
- `mmfi_dataset.py`：新增**根节点中心化**，将绝对世界坐标转换为以 Hip 为原点的相对坐标，解决 MPJPE 与 PA-MPJPE 差距过大的全局位置偏移问题
- `mmfi_dataset.py`：新增**预处理缓存**（`cache_dir`），消除 CPU IO 瓶颈，GPU 利用率从 <5% 提升至 70%+
- `losses.py`：新增**骨骼长度一致性损失**（`L_bone`），约束预测骨架的结构合理性
- `utils/checkpoint.py`：支持**部分权重复用**（`strict=False`），新旧模型结构不匹配时不报错，打印详细加载报告
- `test_single.py` / `test_cross_domain.py`：测试脚本加载权重统一改为 `--work` 参数，`--resume` 仅保留在 `train.py` 用于中断恢复

**v1**
- 初始版本：GAT + CrossSensorAttention + TemporalTransformer 编码器
- 对抗域自适应（GRL + DomainDisentangle + 正交损失）
- 修复原始代码中 domain loss 空操作、梯度反转层缺失、维度硬编码等 4 个 critical bug

---

## 常见问题

**Q: MPJPE 很高但 PA-MPJPE 正常，是什么原因？**

这是**全局位置偏移**问题。PA-MPJPE 在计算前会对齐全局位置，如果两者差距很大（如 MPJPE=344mm，PA-MPJPE=108mm），说明模型学到了骨架形状，但预测的绝对位置偏移很大。根本原因是 CSI 信号本身不包含房间绝对坐标信息。本项目已通过**根节点中心化**解决此问题——训练和测试均在相对坐标空间进行，MPJPE 应接近 PA-MPJPE 量级。

**Q: GPU 利用率很低（< 10%），GPU 几乎空转？**

数据加载是瓶颈，每帧需要实时读取 `.mat` 文件并执行 CSI 预处理。解决方法是开启预处理缓存：

```yaml
# config/default.yaml
data:
  cache_dir: /path/to/MMFi_cache
```

首次运行会自动构建缓存（约需 10-30 分钟），之后 GPU 利用率可从 <5% 提升至 70%+。

**Q: 训练时出现 NaN loss？**

loss 出现 NaN 通常由 CSI 数据中的异常值引起。代码中已内置 NaN batch 跳过机制，并在数据加载时做了 `nan_to_num` 清理。如果频繁出现，可以：
- 适当减小学习率（`--lr 5e-5`）
- 检查 `.mat` 文件是否有损坏帧

**Q: 显存不足（OOM）？**

GAT 层的全节点注意力是主要显存消耗点。可通过以下方式缓解：
- 减小 `batch_size`（`--batch_size 32`）
- 在 `mmfi_dataset.py` 中加大子载波降采样倍率（`::4` 替换 `::2`，N 从 57→28）
- 减小 `model.dim`（`dim: 32`）

**Q: t-SNE 运行很慢？**

默认每域采样 2000 个样本，如需加速可减小：

```bash
python run_all_vis.py ... --n_tsne_samples 500
```

**Q: 如何添加新的源域/目标域组合？**

通过命令行直接指定，无需修改代码：

```bash
python train.py --source E01 E02 --target E03
python test_cross_domain.py --work best.pth --target E03
```

**Q: 旧版 checkpoint 能用于新版代码吗？**

可以部分复用。`utils/checkpoint.py` 的 `load_checkpoint` 默认使用 `strict=False`，能匹配的层直接加载，新增或 shape 变化的层随机初始化，并打印详细的加载报告。但注意：旧版训练时未做根节点中心化，坐标系不同，建议从头重新训练以获得最佳效果。