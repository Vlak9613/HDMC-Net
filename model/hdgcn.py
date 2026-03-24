"""
HDGCN (Hybrid Dynamic Graph Convolutional Network) - Main model for online action recognition.

Uses HDGC (Hybrid Dynamic Graph Convolution) and MomentumNet Extrapolator.
"""

import numpy as np
import math
import torch
import torch.nn.functional as F

from torch import nn

from model.layers import TemporalEncoder, GCN, HDGC
from model.model_utils import import_class
from model.extrapolator import MomentumNetExtrapolator

from einops import rearrange, repeat


class HDGCN(nn.Module):
    """
    HDGCN: Hybrid Dynamic Graph Convolutional Network
    
    Architecture:
    =============
    Input → JointEmbedding → TemporalEncoder(HDGC) → MomentumNet → Recon/Classification
    
    Components:
    ===========
    1. Joint Embedding: Projects 3D coordinates to high-dimensional space
    2. Temporal Encoder: Transformer with HDGC-based attention
    3. MomentumNet: Physics-guided extrapolator for future prediction
    4. Reconstruction Decoder: HDGC-based decoder for skeleton reconstruction
    5. Classification Decoder: HDGC-based decoder for action classification
    """
    def __init__(self, num_class=60, num_point=25, num_person=2,
                 graph=None, in_channels=3, num_head=3, k=0, base_channel=64, depth=4, device='cuda',
                 T=64, n_step=1, dilation=1, num_cls=10, dropout=0.1):
        super(HDGCN, self).__init__()

        self.Graph = import_class(graph)()
        with torch.no_grad():
            A = np.stack([self.Graph.A_norm] * num_head, axis=0)
            self.T = T
            self.arange = torch.arange(T).view(1, 1, T) + 1
            if n_step > 0:
                shift_idx = torch.arange(0, T, dtype=int).view(1, T, 1, 1)
                shift_idx = repeat(shift_idx, 'b t c v -> n b t c v', n=n_step)
                shift_idx = shift_idx - torch.arange(1, n_step+1, dtype=int).view(n_step, 1, 1, 1, 1)
                self.mask = torch.triu(torch.ones(n_step, T), diagonal=1).view(n_step, 1, T, 1, 1).cuda()
                self.shift_idx = shift_idx.cuda() % T
                self.arange_n_step = torch.arange(n_step+1).cuda()
            self.num_class = num_class
            self.num_point = num_point
            self.num_person = num_person
            self.cls_idx = [int(math.ceil(T*i/num_cls)) for i in range(num_cls+1)]
            self.cls_idx[0] = 0
            self.cls_idx[-1] = T
            self.zero = torch.tensor(0.0).cuda()
            self.n_step = n_step
            
        # Joint embedding layer
        self.to_joint_embedding = nn.Linear(in_channels, base_channel)
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_point, base_channel))
        
        # Temporal Encoder with HDGC
        self.temporal_encoder = TemporalEncoder(
            seq_len=T,
            dim=base_channel,
            depth=depth,
            heads=4,
            mlp_dim=base_channel*2,
            dim_head=base_channel//4,
            dropout=dropout,
            emb_dropout=dropout,
            device=device,
            A=A,
            num_point=num_point,
        )

        # MomentumNet Extrapolator
        if n_step > 0:
            self.diffeq_solver = MomentumNetExtrapolator(
                dim=base_channel,
                A=self.Graph.A_norm,
                n_step=n_step,
                T=T,
                V=num_point,
                dropout=dropout,
                num_head=num_head
            ).to(device)
            print(f"[HDMC-Net] Using MomentumNet Extrapolator with HDGC (n_step={n_step})")
        else:
            self.diffeq_solver = None
            print(f"[HDMC-Net] No extrapolation (n_step=0, ablation mode)")

        # Reconstruction decoder with HDGC
        self.recon_decoder = nn.Sequential(
            HDGC(base_channel, base_channel, A, use_gate=True, use_multiscale=True),
            HDGC(base_channel, base_channel, A, use_gate=True, use_multiscale=True),
            nn.Conv2d(base_channel, 3, 1),
        )

        if n_step:
            in_dim = base_channel * (n_step + 1)
            mid_dim = base_channel * (n_step + 1) // 2
            out_dim = base_channel * (n_step + 1) // 2
        else:
            in_dim = base_channel
            mid_dim = base_channel
            out_dim = base_channel

        # Classification decoder with HDGC
        self.cls_decoder = nn.Sequential(
            HDGC(in_dim, mid_dim, A, use_gate=True, use_multiscale=True),
            HDGC(mid_dim, out_dim, A, use_gate=True, use_multiscale=True),
        )

        # Classifiers for different observation rates
        self.c0 = nn.Conv1d(out_dim, num_class, 1)
        self.c1 = nn.Conv1d(out_dim, num_class, 1)
        self.c2 = nn.Conv1d(out_dim, num_class, 1)
        self.c3 = nn.Conv1d(out_dim, num_class, 1)
        self.c4 = nn.Conv1d(out_dim, num_class, 1)
        self.c5 = nn.Conv1d(out_dim, num_class, 1)
        self.c6 = nn.Conv1d(out_dim, num_class, 1)
        self.c7 = nn.Conv1d(out_dim, num_class, 1)
        self.c8 = nn.Conv1d(out_dim, num_class, 1)
        self.c9 = nn.Conv1d(out_dim, num_class, 1)
        self.num_cls = num_cls
        self.classifier_lst = [self.c0, self.c1, self.c2, self.c3, self.c4, 
                               self.c5, self.c6, self.c7, self.c8, self.c9]
        self.spatial_pooling = torch.mean

    def extrapolate(self, z_0, t):
        """Extrapolate future states using MomentumNet."""
        B, C, T, V = z_0.size()
        z_0 = rearrange(z_0, "b c t v -> (b t) c v")

        zs = self.diffeq_solver(z_0, t)
        zs = rearrange(zs, 'n (b t) c v -> n b t c v', t=T)
        z_hat = zs[1:]
        z_hat_shifted = torch.gather(z_hat.clone(), dim=2, index=self.shift_idx.expand_as(z_hat).long())
        z_hat_shifted = self.mask * z_hat_shifted
        z_hat_shifted = rearrange(z_hat_shifted, 'n b t c v -> (n b) c t v')
        z_hat = rearrange(z_hat, 'n b t c v -> (n b) c t v')
        z_0 = rearrange(z_0, '(b t) c v -> b c t v', t=T)

        return z_0, z_hat, z_hat_shifted

    def get_A(self, k):
        """Get adjacency matrix for k-hop."""
        A_outward = self.Graph.A_outward_binary
        I = np.eye(self.Graph.num_node)
        return torch.from_numpy(I - np.linalg.matrix_power(A_outward, k))

    def forward(self, x):
        N, C, T, V, M = x.size()
        x = rearrange(x, 'n c t v m -> (n m t) v c', n=N, m=M, v=V)

        # Joint embedding
        x = self.to_joint_embedding(x)
        x = x + self.pos_embedding[:, :self.num_point]

        # Temporal encoding with HDGC
        x = rearrange(x, '(n m t) v c -> (n m) c t v', m=M, n=N)
        z = self.temporal_encoder(x)

        if self.n_step > 0:
            # Future extrapolation with MomentumNet
            z_0, z_hat, z_hat_shifted = self.extrapolate(z, self.arange_n_step.to(z.dtype))

            # Skeleton reconstruction
            x_hat = self.recon_decoder(z_hat_shifted)
            x_hat = rearrange(x_hat, '(n m l) c t v -> n l c t v m', m=M, l=1).mean(1)

            # Action classification
            z_hat_cls = rearrange(z_hat, '(n b) c t v -> b (n c) t v', n=self.n_step)
            z_cls = self.cls_decoder(torch.cat([z_0, z_hat_cls], dim=1))
        else:
            # No extrapolation (n_step=0): classification only
            z_0 = z
            z_hat = torch.zeros_like(z_0)
            x_hat = torch.zeros(N, C, T, V, M).to(z.device)
            z_cls = self.cls_decoder(z_0)
        z_cls = rearrange(z_cls, '(n m l) c t v -> (n l) m c t v', m=M, l=1).mean(1)
        z_cls = self.spatial_pooling(z_cls, dim=-1)

        y_lst = []
        for i in range(self.num_cls):
            y_lst.append(self.classifier_lst[i](z_cls[:, :, self.cls_idx[i]:self.cls_idx[i+1]]))
        y = torch.cat(y_lst, dim=-1)
        y = rearrange(y, '(n l) c t -> n l c t', l=1).mean(1)
        
        return y, x_hat, z_0, z_hat, self.zero

    def get_attention(self):
        """Get attention maps from temporal encoder."""
        return self.temporal_encoder.get_attention()
