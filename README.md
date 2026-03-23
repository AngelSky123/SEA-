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
L_total = L_pose + 0.1 × L_align + 0.01 × L_domain + 0.01 × L_orth
```

| 损失项 | 说明 |
|--------|------|
| `L_pose` | MSE 姿态回归损失（源域第 0 帧） |
| `L_align` | 源/目标域共享特征均值 MSE 对齐 |
| `L_domain` | 对抗域分类交叉熵（源域=0，目标域=1，经过 GRL） |
| `L_orth` | shared / private 特征正交约束 |

---

## 项目结构

```
.
├── config/
│   ├── default.yaml            # 默认超参数配置
│   └── exp_cross_domain.yaml   # 跨域实验配置
├── dataset/
│   └── mmfi_dataset.py         # MMFi 数据集加载（CSI 预处理、节点注意力聚合）
├── losses/
│   └── losses.py               # 联合损失函数
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
│   ├── checkpoint.py           # checkpoint 保存与加载
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
pip install pyyaml
```

**GPU 建议：** 训练需要至少 12GB 显存（batch_size=8，dim=64）。

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

---

## 快速开始

### 训练（默认配置）

```bash
python train.py
```

### 跨域测试

```bash
python test_cross_domain.py --resume ./outputs/<exp_name>/best.pth
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
[Epoch 010][0020/1800] loss=0.0423  pose=0.0312  align=0.0089  domain=0.6821  orth=0.0043  alpha=0.200
 Epoch 010 done | avg_loss=0.0401 | pose=0.0298 | align=0.0091 | domain=0.6734 | orth=0.0041
  → new best: 0.0401
```

---

## 测试与评估

### 跨域批量测试

```bash
python test_cross_domain.py \
    --resume ./outputs/<exp_name>/best.pth \
    --target E04
```

输出示例：

```
===== RESULTS =====
MPJPE:    0.0842
PA-MPJPE: 0.0631
PCK@0.05: 0.7245
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
  root: /path/to/MMFi   # 数据集根目录
  seq_len: 20           # 输入时间窗口长度（帧数）

train:
  batch_size: 8         # 每批样本数（显存不足时调小）
  epochs: 50            # 训练总轮数
  lr: 0.0001            # Adam 初始学习率（Cosine 衰减至 1e-6）

model:
  dim: 64               # 编码器隐层维度（GAT / Transformer / PoseHead 统一）
  num_joints: 17        # 骨架关节数（H36M 标准）

domain:
  source: [E01, E02, E03]   # 源域环境列表（有标注）
  target: [E04]             # 目标域环境列表（无标注）

log:
  save_dir: ./outputs   # 实验输出根目录
  print_freq: 20        # 每隔多少 iter 打印一次日志
```

**命令行参数优先级：** 命令行 > `--exp` 实验配置 > `--config` 基础配置

---

## 评估指标

| 指标 | 说明 | 越小越好 |
|------|------|---------|
| **MPJPE** | 均关节位置误差，所有关节预测坐标与真值的平均欧氏距离（mm） | ✓ |
| **PA-MPJPE** | Procrustes 对齐后的 MPJPE，消除全局旋转/平移/缩放影响，衡量姿态形状精度 | ✓ |
| **PCK@0.05** | 关节误差 < 0.05m 的比例，衡量粗粒度准确率 | 越大越好 |

---

## 常见问题

**Q: 训练时出现 NaN loss？**

loss 出现 NaN 通常由 CSI 数据中的异常值引起。代码中已内置 NaN batch 跳过机制，并在数据加载时做了 `nan_to_num` 清理。如果频繁出现，可以：
- 适当减小学习率（`--lr 5e-5`）
- 检查 `.mat` 文件是否有损坏帧

**Q: 显存不足（OOM）？**

GAT 层的全节点注意力是主要显存消耗点。可通过以下方式缓解：
- 减小 `batch_size`（`--batch_size 4`）
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
python test_cross_domain.py --resume best.pth --target E03
```
