# HDMC-Net

### A Proactive Online Skeleton Early-Action Intent Recognition Network for Human-Robot Collaborative Assembly Scenarios

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-1.10+-ee4c2c.svg" alt="PyTorch">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
</p>

This is the official implementation of the paper:

> **HDMC-Net: A Proactive Online Skeleton Early-Action Intent Recognition Network for Human-Robot Collaborative Assembly Scenarios**

## Overview

HDMC-Net (**H**ybrid **D**ynamic **M**omentum **C**ausal Network) is a proactive framework for online early skeleton-based action recognition, specifically designed for human-robot collaborative (HRC) assembly scenarios. The network observes incomplete skeleton sequences and predicts the ongoing action *before* it is fully performed, enabling proactive robot collaboration.

### Key Contributions

1. **HDGC (Hybrid Dynamic Graph Convolution):** A novel graph convolution module that fuses learnable prior topology with sample-adaptive topology via channel-wise attention, augmented by multi-scale (1-hop + 2-hop) adjacency and gated stabilization.

2. **MomentumNet Extrapolator:** A physics-guided future prediction module that uses velocity-based extrapolation with learnable corrections, ensuring non-trivial predictions even in early observation stages.

3. **Progressive Attention Bias (PAB):** A causal temporal encoding mechanism with learnable relative position bias that enforces temporal causality while progressively weighting recent frames.

4. **Multi-Loss Training:** A joint optimization framework combining classification, reconstruction, feature consistency, and classification guidance losses.

## Architecture

```
Input [N, 3, T, V, M]
        │
        ▼
┌─────────────────┐
│ Joint Embedding  │  3D → 64D (Linear + Positional Encoding)
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────┐
│     Temporal Encoder (HDGC)         │
│  ─ Transformer with HDGC Attention  │
│  ─ Progressive Attention Bias (PAB) │
│  ─ Causal Mask                      │
└────────────────┬────────────────────┘
                 │ z_0
                 ▼
┌─────────────────────────────────────┐
│      MomentumNet Extrapolator       │
│  ─ Physics-guided prediction        │
│  ─ Velocity-based extrapolation     │
│  ─ Learnable velocity correction    │
└────────────────┬────────────────────┘
                 │ z_hat
       ┌─────────┴─────────┐
       ▼                   ▼
┌─────────────┐     ┌─────────────┐
│Recon Decoder│     │ Cls Decoder  │
│  (HDGC ×2)  │     │  (HDGC ×2)  │
└──────┬──────┘     └──────┬──────┘
       │                   │
       ▼                   ▼
   x_hat (recon)      y_hat [N, C, T] (classification)
```

### HDGC — Hybrid Dynamic Graph Convolution

```
A_final = A_prior + λ · A_adapt + β · A_2hop
```

| Feature | Description |
|---------|-------------|
| **Hybrid Topology Fusion** | Learnable prior topology + sample-adaptive attention topology |
| **Multi-Scale Adjacency** | 1-hop direct + 2-hop indirect connections |
| **Channel-wise Attention** | Query-Key mechanism to dynamically generate graph structure |
| **Gated Stabilization** | Sigmoid gating to prevent gradient explosion |
| **Multi-Head Design** | Parallel learning of multiple relation patterns |

### MomentumNet Extrapolator

```
v     = z[:, :, -1] - z[:, :, -2]      # Velocity from data (guaranteed non-zero)
Δv    = Network(z, v_encoded)           # Learned velocity correction (via HDGC)
z_next = z_last + (decay · v + Δv) · dt # Physics-guided extrapolation
```

## Project Structure

```
HDMC-Net/
├── train.py                # Training & evaluation entry point
├── config.py               # Command-line argument definitions
├── losses.py               # Loss functions (Label Smoothing CE, Masked MSE)
├── utils.py                # Utilities (skeleton alignment, metrics, etc.)
├── visualize.py            # Visualization tools
│
├── model/
│   ├── hdgcn.py            # Main HDGCN model
│   ├── layers.py           # Core layers (HDGC, GCN, TemporalEncoder, Attention)
│   ├── extrapolator.py     # MomentumNet Extrapolator
│   └── model_utils.py      # Model initialization utilities
│
├── feeders/
│   ├── ntu_feeder.py       # NTU RGB+D data loader
│   ├── ucla_feeder.py      # NW-UCLA data loader
│   ├── hrc_feeder.py       # HRC-Assembly data loader
│   └── feeder_utils.py     # Data augmentation utilities
│
├── graph/
│   ├── ntu_graph.py        # NTU skeleton graph (25 joints)
│   ├── ucla_graph.py       # UCLA skeleton graph (20 joints)
│   └── graph_utils.py      # Graph construction utilities
│
└── data/
    ├── nturgbd_raw/        # NTU RGB+D raw skeleton files (.skeleton)
    ├── ntu/                # NTU RGB+D 60 processed data
    │   ├── statistics/     # Metadata (labels, cameras, performers)
    │   ├── get_raw_skes_data.py
    │   ├── get_raw_denoised_data.py
    │   └── seq_transformation.py
    ├── NW-UCLA/            # NW-UCLA dataset
    │   └── all_sqe/        # Raw data (.json)
    └── hrc/                # HRC-Assembly dataset
        ├── statistics/     # Metadata
        ├── build_dataset.py
        └── seq_transformation.py
```

## Installation

### Requirements

- Python ≥ 3.8
- PyTorch ≥ 1.10
- CUDA ≥ 11.0

### Setup

```bash
conda create -n hdmc python=3.8
conda activate hdmc

pip install torch torchvision
pip install einops tqdm wandb h5py
pip install numpy scipy scikit-learn
pip install matplotlib seaborn pandas
```

## Data Preparation

### Supported Datasets

| Dataset | Classes | Joints | Persons | Evaluation Protocol |
|---------|---------|--------|---------|---------------------|
| NTU RGB+D 60 | 60 | 25 | 2 | Cross-Subject (CS) / Cross-View (CV) |
| NW-UCLA | 10 | 20 | 1 | Cross-View |
| HRC-Assembly | 17 | 25 | 1 | Cross-Subject (CS) |

---

### 1. NTU RGB+D 60

#### Download

1. Request and download the dataset from [ROSE Lab](https://rose1.ntu.edu.sg/dataset/actionRecognition).
2. Download the skeleton data: `nturgbd_skeletons_s001_to_s017.zip` (NTU RGB+D 60).
3. Extract to `./data/nturgbd_raw/nturgb+d_skeletons/`.

#### Preprocessing

Run the following three steps sequentially in the `data/ntu/` directory:

```bash
cd ./data/ntu

# Step 1: Extract raw skeleton data from .skeleton files
python get_raw_skes_data.py

# Step 2: Denoise — filter noisy skeletons via length/spread/motion heuristics
python get_raw_denoised_data.py

# Step 3: Sequence transformation — centering, frame alignment, dataset split, and skeleton alignment
python seq_transformation.py
```

This will generate:

| File | Description |
|------|-------------|
| `NTU60_CS.npz` | Cross-Subject split (unaligned) |
| `NTU60_CV.npz` | Cross-View split (unaligned) |
| `NTU60_CS_aligned.npz` | Cross-Subject split (skeleton-aligned) |
| `NTU60_CV_aligned.npz` | Cross-View split (skeleton-aligned) |

Create symbolic links for training:

```bash
cd ./data/ntu
ln -sf NTU60_CS_aligned.npz CS_aligned.npz
ln -sf NTU60_CV_aligned.npz CV_aligned.npz
```

#### Data Format

The `*_aligned.npz` files contain:

| Field | Shape | Description |
|-------|-------|-------------|
| `x_train` | (N_train, 300, 150) | Training skeleton sequences; 150 = 2 persons × 25 joints × 3 coords |
| `y_train` | (N_train, 60) | Training labels (one-hot) |
| `x_test` | (N_test, 300, 150) | Test skeleton sequences |
| `y_test` | (N_test, 60) | Test labels (one-hot) |

| Split | Train Samples | Test Samples |
|-------|---------------|--------------|
| Cross-Subject (CS) | 40,091 | 16,487 |
| Cross-View (CV) | 37,646 | 18,932 |

---

### 2. NW-UCLA

1. Download the NW-UCLA dataset from [CTR-GCN](https://github.com/Uason-Chen/CTR-GCN).
2. Place the `all_sqe` folder into `./data/NW-UCLA/`.

No additional preprocessing is needed — the UCLA feeder reads JSON files directly.

---

### 3. HRC-Assembly

The HRC-Assembly dataset is collected using Azure Kinect for human-robot collaborative assembly scenarios.

| Property | Value |
|----------|-------|
| Subjects | 9 (P01–P09) |
| Action classes | 17 (A01–A17) |
| Repetitions | 10 per action per subject |
| Total samples | 1,530 |
| Joints | 25 (Azure Kinect body tracking) |
| Coordinate unit | Millimeters |

#### Preprocessing

```bash
cd ./data/hrc

# Step 1: Build intermediate data from raw .npy skeleton files
python build_dataset.py --raw_dir ./raw_skeletons

# Step 2: Sequence transformation — centering, frame alignment, cross-subject split, skeleton alignment
python seq_transformation.py
```

Create symbolic link:

```bash
cd ./data/hrc
ln -sf HRC_CS_aligned.npz CS_aligned.npz
```

#### Cross-Subject Split

| Set | Subjects | Samples |
|-----|----------|---------|
| Train | P01–P06 | 1,020 |
| Test | P07–P09 | 510 |

## Training

### NTU RGB+D 60 — Cross-Subject

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
    --n_step=6 \
    --num_cls=10 \
    --dropout=0.1
```

### NTU RGB+D 60 — Cross-View

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
    --datacase=CV \
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
    --n_step=6 \
    --num_cls=10 \
    --dropout=0.1
```

### NW-UCLA

```bash
python train.py \
    --half=True \
    --batch_size=32 \
    --test_batch_size=64 \
    --step 50 60 \
    --num_epoch=70 \
    --num_worker=4 \
    --dataset=ucla \
    --num_class=10 \
    --datacase=ucla \
    --weight_decay=0.0005 \
    --num_person=1 \
    --num_point=20 \
    --graph=graph.ucla_graph.Graph \
    --feeder=feeders.ucla_feeder.Feeder \
    --base_lr=0.1 \
    --base_channel=64 \
    --window_size=64 \
    --lambda_1=1.0 \
    --lambda_2=0.1 \
    --lambda_3=0.1 \
    --lambda_cls_guide=0.1 \
    --n_step=6 \
    --num_cls=10 \
    --dropout=0.1
```

### HRC-Assembly

```bash
python train.py \
    --half=True \
    --batch_size=32 \
    --test_batch_size=64 \
    --step 50 60 \
    --num_epoch=70 \
    --num_worker=4 \
    --dataset=hrc \
    --num_class=17 \
    --datacase=CS \
    --weight_decay=0.0005 \
    --num_person=1 \
    --num_point=25 \
    --graph=graph.ntu_graph.Graph \
    --feeder=feeders.hrc_feeder.Feeder \
    --base_lr=0.1 \
    --base_channel=64 \
    --window_size=64 \
    --lambda_1=1.0 \
    --lambda_2=0.1 \
    --lambda_3=0.1 \
    --lambda_cls_guide=0.1 \
    --n_step=6 \
    --num_cls=10 \
    --dropout=0.1
```

## Testing

To evaluate a trained model:

```bash
python train.py \
    --phase=test \
    --weights=<path_to_checkpoint.pt> \
    --dataset=ntu \
    --datacase=CS \
    --num_class=60 \
    --num_person=2 \
    --num_point=25 \
    --graph=graph.ntu_graph.Graph \
    --feeder=feeders.ntu_feeder.Feeder \
    --window_size=64 \
    --n_step=6 \
    --num_cls=10
```

## Loss Functions

HDMC-Net employs a multi-loss training objective:

| Loss | Formula | Weight | Description |
|------|---------|--------|-------------|
| **Classification** | Label Smoothing Cross Entropy | `λ₁ = 1.0` | Main classification loss on predicted action labels |
| **Reconstruction** | Masked MSE | `λ₂ = 0.1` | Reconstruction of future skeleton frames from extrapolated features |
| **Feature Consistency** | MSE | `λ₃ = 0.1` | Alignment between encoder outputs and extrapolated features |
| **Classification Guidance** | Cross Entropy | `λ_cg = 0.1` | Auxiliary classification on extrapolated representations |

```
L_total = λ₁ · L_cls + λ₂ · L_recon + λ₃ · L_feature + λ_cg · L_cls_guide
```

## Key Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--dataset` | `ntu` | Dataset name (`ntu`, `ucla`, `hrc`) |
| `--datacase` | `CS` | Evaluation protocol (`CS`, `CV`, `ucla`) |
| `--num_class` | `60` | Number of action classes |
| `--num_point` | `25` | Number of skeleton joints |
| `--num_person` | `2` | Number of persons per sample |
| `--batch_size` | `64` | Training batch size |
| `--base_lr` | `0.1` | Initial learning rate |
| `--num_epoch` | `110` | Total training epochs |
| `--step` | `[50, 60]` | Learning rate decay milestones |
| `--n_step` | `3` | MomentumNet extrapolation steps |
| `--num_cls` | `10` | Number of temporal segment classifiers |
| `--base_channel` | `64` | Base hidden channel dimension |
| `--depth` | `4` | Transformer encoder depth |
| `--n_heads` | `3` | Number of graph convolution heads |
| `--dropout` | `0.1` | Dropout rate |
| `--lambda_1` | `1.0` | Classification loss weight |
| `--lambda_2` | `0.1` | Reconstruction loss weight |
| `--lambda_3` | `0.01` | Feature consistency loss weight |
| `--lambda_cls_guide` | `0.05` | Classification guidance loss weight |
| `--half` | `True` | Enable FP16 mixed-precision training |
| `--window_size` | `64` | Temporal window size |
| `--random_rot` | `True` | Random rotation augmentation |

## Acknowledgements

This project is built upon the following open-source works:

- [InfoGCN++](https://github.com/stnoah1/infogcnpp) — Online skeleton-based action recognition framework
- [InfoGCN](https://github.com/stnoah1/infogcn) — Information bottleneck graph convolutional network
- [2s-AGCN](https://github.com/lshiwjx/2s-AGCN) — Two-stream adaptive graph convolutional network
- [CTR-GCN](https://github.com/Uason-Chen/CTR-GCN) — Channel-wise topology refinement graph convolution


## Citation

If you find this work useful for your research, please cite:

```bibtex
@article{hdmcnet2026,
  title={HDMC-Net: A Proactive Online Skeleton Early-Action Intent Recognition Network for Human-Robot Collaborative Assembly Scenarios},
  author={},
  journal={},
  year={2026}
}
```

## License

This project is licensed under the [MIT License](LICENSE).
