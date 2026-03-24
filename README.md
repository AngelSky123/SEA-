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
- **时序 Transformer**（含位置编码）：在时间维度上捕捉动作的连续性，每个天线节点独立建模

---

## 方法概述

```
CSI Input (B, T, N, C)
        │
        ▼
┌───────────────────┐
│  GATLayer          │  ← 空间传感器间图注意力
│  CrossSensorAttn   │  ← 跨传感器自注意力
│  TemporalTF + PE   │  ← 时序 Transformer（每个节点独立，保留节点多样性）
└───────────────────┘
        │  Encoder Output (B, T, N, D)
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
> 从而习得域不变表示。对 private 特征做域分类没有对抗意义。

**损失函数：**

```
L_total = L_pose + 0.1 × L_vel + 0.1 × L_bone + 0.01 × L_align + 0.01 × L_domain + 0.01 × L_orth
```

| 损失项 | 权重 | 说明 |
|--------|------|------|
| `L_pose` | 1.0 | MSE 姿态回归损失，对根节点中心化后的相对坐标监督 |
| `L_vel` | 0.1 | 帧间位移监督（末帧 - 首帧），提供运动方向信号，权重较低避免喧宾夺主 |
| `L_bone` | 0.1 | 骨骼长度一致性损失，约束预测骨架各骨骼长度与 GT 一致 |
| `L_align` | 0.01 | 源/目标域 shared 特征均值 MSE 对齐 |
| `L_domain` | 0.01 | 对抗域分类交叉熵（源域=0，目标域=1，经过 GRL，作用于 shared 特征） |
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
│   ├── attention.py            # CrossSensorAttention（注意力权重缓存）
│   ├── disentangle.py          # DomainDisentangle（shared/private 特征分离）
│   ├── domain_disc.py          # 域判别器（内置 GRL，作用于 shared 特征）
│   ├── encoder.py              # 三阶段编码器（GAT + Attn + Transformer）
│   ├── gat.py                  # 图注意力层
│   ├── grad_reverse.py         # 梯度反转层（GRL）
│   ├── model.py                # WiFiPoseModel 主模型
│   ├── pose_head.py            # 姿态回归头
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
[Epoch 010][0020/3505] loss=0.0312  pose=0.0198  vel=0.0041  bone=0.0087  align=0.0003  domain=1.3821  orth=0.0000  alpha=0.200
 Epoch 010 done | avg=0.0301 | pose=0.0192 | vel=0.0038 | bone=0.0083 | align=0.0003 | domain=1.3756 | orth=0.0000
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

**v3（当前版本）**

> 本次更新修复了 v2 中存在的若干 critical bug 及设计问题，建议所有用户更新。

**Critical 修复：**

- `utils/checkpoint.py`：`load_state_dict` 增加 `strict=False`，修复 README 声明与代码实现不符的问题。现在会打印详细的 missing/unexpected key 报告，新旧结构不匹配时不再直接报错
- `visualization/attention_vis.py`：修复 `gat.fc` 不存在（应为 `gat.proj`）导致的 `AttributeError`；修复手动复现 encoder 调用链时注意力权重未被写入缓存的问题，改为直接调用 `model.encoder(x)` 触发完整前向；修复 DataLoader 二元组解包错误
- `visualization/experiment_report.py`：修复 DataLoader 二元组解包错误（`MMFiDataset` 返回三元组）
- `visualization/joint_error_analysis.py`：修复 DataLoader 二元组解包错误
- `visualization/tsne_analysis.py`：修复 DataLoader 二元组解包错误

**Design 修复：**

- `models/model.py`：域判别器改为作用于 **shared** 特征（原版作用于 private 特征，缺乏对抗意义）。正确做法是对 shared 特征施加 GRL，迫使 encoder 产生域不变的 shared 表示
- `models/transformer.py`：`TemporalTransformer` 改为对每个节点独立建模（`(B*N, T, D)`），原版先对节点做均值再广播，导致所有节点在 Temporal 之后特征完全相同，破坏了后续 CrossSensorAttention 的节点多样性
- `losses/losses.py`：帧间位移监督语义明确为"末帧相对首帧的关节位移"，权重从 0.5 降至 0.1，避免与 `L_pose` 量级相近时主导梯度

**Minor 修复：**

- `visualization/attention_vis.py`、`experiment_report.py`、`joint_error_analysis.py`、`tsne_analysis.py`：`--dim` 默认值从 64 统一修正为 256，与 `config/default.yaml` 保持一致
- `dataset/mmfi_dataset.py`：不再在预处理阶段对天线维度做加权求和（原版输出 `(P, 4)` 丢失空间信息）。修复后保留 `(N, P, 4)` 形状，展平为 `(N, P*C)` 输入模型，让 GAT/CrossSensorAttention 自行学习天线间关系。旧行为可通过 `keep_spatial=False` 恢复
- `visualization/plot_loss.py`：将 `plt.show()` 改为 `plt.savefig()`，避免在服务器环境挂起；添加非交互式 Agg 后端声明，与项目其他可视化模块风格统一

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

**Q: MPJPE 很高但 PA-MPJPE 正常，是什么原因？**

这是**全局位置偏移**问题。PA-MPJPE 在计算前会对齐全局位置，如果两者差距很大（如 MPJPE=344mm，PA-MPJPE=108mm），说明模型学到了骨架形状，但预测的绝对位置偏移很大。本项目已通过**根节点中心化**解决此问题——训练和测试均在相对坐标空间进行，MPJPE 应接近 PA-MPJPE 量级。

**Q: GPU 利用率很低（< 10%）？**

开启预处理缓存：

```yaml
# config/default.yaml
data:
  cache_dir: /path/to/MMFi_cache
```

首次运行会自动构建缓存（约需 10-30 分钟），之后 GPU 利用率可从 <5% 提升至 70%+。

**Q: 训练时出现 NaN loss？**

代码已内置 NaN batch 跳过机制，并在数据加载时做了 `nan_to_num` 清理。如果频繁出现，可以：
- 适当减小学习率（`--lr 5e-5`）
- 检查 `.mat` 文件是否有损坏帧

**Q: 显存不足（OOM）？**

- 减小 `batch_size`（`--batch_size 32`）
- 减小 `model.dim`（`dim: 128`）
- 在 `mmfi_dataset.py` 中加大子载波降采样倍率（`::4` 替换 `::2`）

**Q: 可视化脚本运行时报 "too many values to unpack"？**

这是 v2 及更早版本的已知 bug。`MMFiDataset.__getitem__` 返回三元组 `(csi, pose, root_offset)`，但旧版可视化脚本用二元组解包。**v3 已全部修复**，请更新所有可视化脚本。

**Q: 旧版 checkpoint 能用于 v3 代码吗？**

可以。`utils/checkpoint.py` 使用 `strict=False` 加载，能匹配的层直接复用，并打印详细报告。

但需注意两点：
1. v3 修改了 `TemporalTransformer` 的节点建模方式（`(B*N, T, D)` 而非均值后 expand），导致 `transformer.encoder` 的权重无法直接复用（节点数 N 不同），会随机初始化该模块；
2. v3 修改了 `mmfi_dataset.py` 的 CSI 输出维度（保留天线空间信息），若旧模型的 `encoder.gat.proj` 输入维度不匹配，该层也会随机初始化。

建议从头训练以获得最佳效果。

**Q: t-SNE 运行很慢？**

```bash
python run_all_vis.py ... --n_tsne_samples 500
```

**Q: 如何添加新的源域/目标域组合？**

```bash
python train.py --source E01 E02 --target E03
python test_cross_domain.py --work best.pth --target E03
```

**Q: domain loss 一直很高（远大于 1.386）？**

说明域判别器仍能轻松区分源域与目标域，域对齐尚未成功。可以：
- 增大训练 epoch（GRL 的 alpha 随 epoch 线性增大，需要足够轮数才能充分对抗）
- 检查 `model.py` 中域判别器是否正确接在 **shared** 特征上（v3 已修复，旧版接在 private 上无对抗效果）
- 尝试增大 `domain loss` 权重（`0.01 → 0.05`）