# HDMC-Net: Hybrid Dynamic Momentum Causal Network

**混合动态动量因果网络** - 用于在线早期骨架动作识别

## 核心创新

### 1. HDGC (Hybrid Dynamic Graph Convolution) - 混合动态图卷积

```
A_final = A_prior + λ * A_adapt + β * A_2hop
```

| 特性 | 描述 |
|------|------|
| **混合拓扑融合** | 先验拓扑 + 自适应拓扑动态平衡 |
| **多尺度邻接** | 1-hop直连 + 2-hop间接连接 |
| **通道注意力** | Query-Key机制动态生成图结构 |
| **门控稳定** | Sigmoid门控防止梯度爆炸 |
| **多头设计** | 并行学习多种关系模式 |

### 2. MomentumNet Extrapolator - 动量网络外推器

```
v = z[:, :, -1] - z[:, :, -2]      # 数据速度
Δv = Network(z, v_encoded)          # 速度修正
z_next = z_last + (v + Δv) * dt     # 外推预测
```

- **物理引导**: 基于速度的预测模拟物理运动
- **非零保证**: 基础速度来自数据，确保有意义的预测
- **可学习动力学**: 网络学习基于上下文的速度修正

## 项目结构

```
HDMC-Net/
├── train.py              # 训练入口
├── config.py             # 命令行配置
├── losses.py             # 损失函数
├── visualize.py          # 可视化工具
├── utils.py              # 通用工具 (含骨架对齐)
│
├── model/
│   ├── hdgcn.py          # 主模型 HDGCN
│   ├── layers.py         # 网络层 (HDGC, GCN, Transformer)
│   ├── extrapolator.py   # MomentumNet 外推器
│   └── model_utils.py    # 模型工具
│
├── feeders/
│   ├── ntu_feeder.py     # NTU RGB+D 数据加载器
│   ├── ucla_feeder.py    # NW-UCLA 数据加载器
│   └── feeder_utils.py   # 数据增强工具
│
├── graph/
│   ├── ntu_graph.py      # NTU 骨架图 (25关节)
│   ├── ucla_graph.py     # UCLA 骨架图 (20关节)
│   └── graph_utils.py    # 图构建工具
│
└── data/
    ├── nturgbd_raw/
    │   └── nturgb+d_skeletons/    # NTU RGB+D 60 原始骨架文件 (.skeleton)
    ├── ntu/
    │   ├── statistics/            # 元数据 (标签、摄像头、表演者等)
    │   ├── get_raw_skes_data.py   # 步骤1: 提取原始骨架数据
    │   ├── get_raw_denoised_data.py # 步骤2: 去噪
    │   ├── seq_transformation.py  # 步骤3: 序列变换 + 生成 npz
    │   ├── NTU60_CS_aligned.npz   # 处理后数据 (Cross-Subject)
    │   ├── NTU60_CV_aligned.npz   # 处理后数据 (Cross-View)
    │   ├── CS_aligned.npz -> NTU60_CS_aligned.npz  # 训练用符号链接
    │   └── CV_aligned.npz -> NTU60_CV_aligned.npz  # 训练用符号链接
    └── NW-UCLA/
        └── all_sqe/               # NW-UCLA 原始数据 (.json)
```

## 安装依赖

```bash
conda create -n ear python=3.8
conda activate ear
pip install torch torchvision
pip install einops tqdm wandb h5py
pip install numpy scipy scikit-learn
pip install matplotlib seaborn pandas
```

## 数据准备

### 支持的数据集

| 数据集 | 类别数 | 关节数 | 评估协议 |
|--------|--------|--------|----------|
| NTU RGB+D 60 | 60 | 25 | Cross-Subject (CS) / Cross-View (CV) |
| NTU RGB+D 120 | 120 | 25 | Cross-Subject (CSub) / Cross-Setup (CSet) |
| NW-UCLA | 10 | 20 | Cross-View |

### 下载数据集

#### NTU RGB+D 60 和 120

1. 前往 [ROSE Lab](https://rose1.ntu.edu.sg/dataset/actionRecognition) 申请并下载数据集
2. 下载骨架数据:
   - `nturgbd_skeletons_s001_to_s017.zip` (NTU RGB+D 60)
   - `nturgbd_skeletons_s018_to_s032.zip` (NTU RGB+D 120, 可选)
3. 解压到 `./data/nturgbd_raw/` 目录下

#### NW-UCLA

1. 从 [CTR-GCN](https://github.com/Uason-Chen/CTR-GCN) 仓库下载 NW-UCLA 数据集
2. 将 `all_sqe` 文件夹放入 `./data/NW-UCLA/` 目录下

### 目录结构

下载完成后，确保数据目录结构如下:

```
data/
├── NW-UCLA/
│   └── all_sqe/
│       ... # NW-UCLA 原始数据 (.json)
├── ntu/
│   └── statistics/           # 元数据文件 (已包含在仓库中)
├── ntu120/                   # (可选, NTU RGB+D 120)
│   └── statistics/
└── nturgbd_raw/
    ├── nturgb+d_skeletons/   # 解压自 nturgbd_skeletons_s001_to_s017.zip
    │   ├── S001C001P001R001A001.skeleton
    │   ├── S001C001P001R001A002.skeleton
    │   └── ... (共 56,880 个文件)
    └── nturgb+d_skeletons120/ # (可选) 解压自 nturgbd_skeletons_s018_to_s032.zip
        └── ...
```

### 处理 NTU RGB+D 60 数据

数据处理分为三步，需在 `data/ntu/` 目录下依次执行:

```bash
cd ./data/ntu

# 步骤1: 从 .skeleton 文件提取原始骨架数据
# 会自动排除 302 个已知的缺失/损坏样本 (见 statistics/samples_with_missing_skeletons.txt)
python get_raw_skes_data.py

# 步骤2: 去噪 - 基于长度、扩散和运动启发式规则过滤噪声骨架
python get_raw_denoised_data.py

# 步骤3: 序列变换 - 中心化、帧对齐、数据集划分，生成最终 .npz 文件
python seq_transformation.py
```

处理完成后将生成以下文件:

| 文件 | 说明 | 大小 |
|------|------|------|
| `NTU60_CS.npz` | Cross-Subject 划分 (未对齐) | ~9.6 GB |
| `NTU60_CV.npz` | Cross-View 划分 (未对齐) | ~9.6 GB |
| `NTU60_CS_aligned.npz` | Cross-Subject 划分 (骨架对齐) | ~9.6 GB |
| `NTU60_CV_aligned.npz` | Cross-View 划分 (骨架对齐) | ~9.6 GB |

> **注意**: 完整处理需要约 40 GB 磁盘空间 (含中间文件)。如果空间不足，可在每步完成后删除 `raw_data/` 和 `denoised_data/` 中间目录，并删除未对齐的 `NTU60_CS.npz` 和 `NTU60_CV.npz`，只保留 `*_aligned.npz` 文件。

最后，在 `data/ntu/` 下创建训练所需的符号链接:

```bash
cd ./data/ntu
ln -sf NTU60_CS_aligned.npz CS_aligned.npz
ln -sf NTU60_CV_aligned.npz CV_aligned.npz
```

### 处理 NTU RGB+D 120 数据 (可选)

```bash
cd ./data/ntu120
python get_raw_skes_data.py
python get_raw_denoised_data.py
python seq_transformation.py
```

### 数据格式说明

训练使用的 `*_aligned.npz` 文件包含以下字段:

| 字段 | 形状 | 说明 |
|------|------|------|
| `x_train` | (N_train, 300, 150) | 训练集骨架序列, 150 = 2人 × 25关节 × 3坐标 |
| `y_train` | (N_train, 60) | 训练集标签 (one-hot) |
| `x_test` | (N_test, 300, 150) | 测试集骨架序列 |
| `y_test` | (N_test, 60) | 测试集标签 (one-hot) |

NTU RGB+D 60 数据规模:

| 协议 | 训练样本 | 测试样本 |
|------|----------|----------|
| Cross-Subject (CS) | 40,091 | 16,487 |
| Cross-View (CV) | 37,646 | 18,932 |

## 训练

### NTU RGB+D Cross-Subject (60类)

```bash
python train.py \
    --half=True \
    --batch_size=32 \
    --test_batch_size=64 \
    --step 50 60 \
    --num_epoch=70 \
    --num_worker=4 \
    --dataset=ntu \
    --num_class=60 \
    --datacase=CS \
    --weight_decay=0.0005 \
    --num_person=2 \
    --num_point=25 \
    --graph=graph.ntu_graph.Graph \
    --feeder=feeders.ntu_feeder.Feeder \
    --base_lr=0.1 \
    --base_channel=64 \
    --window_size=64 \
    --lambda_1=1.0 \
    --lambda_2=0.1 \
    --lambda_3=0.1 \
    --lambda_cls_guide=0.1 \
    --n_step=3 \
    --num_cls=10 \
    --dropout=0.1
```

### NTU RGB+D Cross-View (60类)

```bash
python train.py \
    --dataset=ntu \
    --datacase=CV \
    --num_class=60 \
    --graph=graph.ntu_graph.Graph \
    --feeder=feeders.ntu_feeder.Feeder \
    --batch_size=32 \
    --num_epoch=70
```

### NTU RGB+D 120类

```bash
python train.py \
    --dataset=ntu \
    --datacase=CS \
    --num_class=120 \
    --graph=graph.ntu_graph.Graph \
    --feeder=feeders.ntu_feeder.Feeder \
    --batch_size=32 \
    --num_epoch=70
```

### NW-UCLA 数据集

```bash
python train.py \
    --dataset=ucla \
    --num_class=10 \
    --num_point=20 \
    --graph=graph.ucla_graph.Graph \
    --feeder=feeders.ucla_feeder.Feeder \
    --window_size=52 \
    --batch_size=16 \
    --num_epoch=70
```

## 测试

```bash
python train.py \
    --phase=test \
    --weights=<path_to_weights.pt> \
    --dataset=ntu \
    --datacase=CS \
    --graph=graph.ntu_graph.Graph \
    --feeder=feeders.ntu_feeder.Feeder
```

## 主要参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--dataset` | ntu | 数据集名称 |
| `--datacase` | CS | 数据划分 (CS/CV) |
| `--num_class` | 60 | 动作类别数 |
| `--num_point` | 25 | 骨架关节数 |
| `--batch_size` | 64 | 训练批大小 |
| `--base_lr` | 0.1 | 初始学习率 |
| `--num_epoch` | 110 | 训练轮数 |
| `--n_step` | 3 | MomentumNet预测步数 |
| `--lambda_1` | 1.0 | 分类损失权重 |
| `--lambda_2` | 0.1 | 重建损失权重 |
| `--lambda_3` | 0.01 | 特征一致性损失权重 |
| `--lambda_cls_guide` | 0.05 | 分类引导损失权重 |
| `--base_channel` | 64 | 基础通道数 |
| `--depth` | 4 | Transformer深度 |
| `--dropout` | 0.1 | Dropout率 |
| `--half` | True | 使用FP16训练 |

## 网络架构

```
输入 [N, 3, T, V, M]
        │
        ▼
┌─────────────────┐
│ Joint Embedding │  3D → 64D
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────┐
│     Temporal Encoder (HDGC)         │
│  - Transformer with HDGC Attention  │
│  - Progressive Attention Bias       │
│  - Causal Mask                      │
└────────────────┬────────────────────┘
                 │ z_0
                 ▼
┌─────────────────────────────────────┐
│      MomentumNet Extrapolator       │
│  - Physics-guided prediction        │
│  - Velocity-based extrapolation     │
└────────────────┬────────────────────┘
                 │ z_hat
       ┌─────────┴─────────┐
       ▼                   ▼
┌─────────────┐     ┌─────────────┐
│Recon Decoder│     │ Cls Decoder │
│  (HDGC ×2)  │     │  (HDGC ×2)  │
└──────┬──────┘     └──────┬──────┘
       │                   │
       ▼                   ▼
    x_hat              y_hat [N, C, T]
```

## 损失函数

- **Classification Loss**: Label Smoothing Cross Entropy
- **Reconstruction Loss**: Masked MSE Loss
- **Feature Consistency Loss**: MSE between encoder and extrapolator outputs
- **Classification Guidance Loss**: Cross Entropy on extrapolated features

## 致谢

本项目基于以下开源工作:

- [InfoGCN++](https://github.com/stnoah1/infogcnpp) - 在线骨架动作识别框架
- [InfoGCN](https://github.com/stnoah1/infogcn) - 信息瓶颈图卷积网络
- [2s-AGCN](https://github.com/lshiwjx/2s-AGCN) - 双流自适应图卷积网络
- [CTR-GCN](https://github.com/Uason-Chen/CTR-GCN) - 通道拓扑细化图卷积网络

数据处理代码参考自:
- [SGN](https://github.com/microsoft/SGN)、[HCN](https://github.com/huguyuehuhu/HCN-pytorch)、[Predict & Cluster](https://github.com/shlizee/Predict-Cluster)

## 引用

如果本项目对您的研究有帮助，请引用：

```bibtex
@article{hdmcnet2024,
  title={HDMC-Net: Hybrid Dynamic Momentum Causal Network for Online Early Skeleton-based Action Recognition},
  author={...},
  journal={...},
  year={2024}
}
```

## License

MIT License
