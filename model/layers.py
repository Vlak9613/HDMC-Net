"""
HDMC-Net Network Layers

Core modules:
- HDGC (Hybrid Dynamic Graph Convolution): Main graph convolution with dynamic topology
- GCN: Basic graph convolution for ablation study
- TemporalEncoder: Transformer-based temporal encoder
"""

import math

import torch
import numpy as np

from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange

from model.model_utils import conv_init, bn_init, conv_branch_init


class GCN(nn.Module):
    """Basic Graph Convolution Network for ablation study."""
    def __init__(self, in_channels, out_channels, A):
        super(GCN, self).__init__()
        self.out_c = out_channels
        self.in_c = in_channels
        self.num_head = A.shape[0]
        self.shared_topology = nn.Parameter(torch.from_numpy(A.astype(np.float32)), requires_grad=True)

        self.conv_d = nn.ModuleList()
        for i in range(self.num_head):
            self.conv_d.append(nn.Conv2d(in_channels, out_channels, 1))

        if in_channels != out_channels:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.down = lambda x: x

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        bn_init(self.bn, 1e-6)
        for i in range(self.num_head):
            conv_branch_init(self.conv_d[i], self.num_head)

    def forward(self, x):
        N, C, T, V = x.size()

        out = None
        A = self.shared_topology.unsqueeze(0)
        for h in range(self.num_head):
            A_h = A[:, h, :, :]
            feature = rearrange(x, 'n c t v -> (n t) v c')
            z = A_h @ feature
            z = rearrange(z, '(n t) v c-> n c t v', t=T).contiguous()
            z = self.conv_d[h](z)
            out = z + out if out is not None else z

        out = self.bn(out)
        out += self.down(x)
        out = self.relu(out)

        return out


class HDGC(nn.Module):
    """
    HDGC (Hybrid Dynamic Graph Convolution) - 混合动态图卷积
    
    Core Design:
    ============
    A_final = A_prior + λ * A_adapt + β * A_2hop
    
    Key Features:
    =============
    1. Hybrid Topology Fusion (混合拓扑融合)
       - A_prior: Skeleton prior topology (learnable, initialized from anatomy)
       - A_adapt: Sample-adaptive topology (computed via attention)
       - λ: Learnable fusion weight balancing prior and adaptive
    
    2. Multi-Scale Adjacency (多尺度邻接)
       - 1-hop: Direct physical connections (e.g., wrist-elbow)
       - 2-hop: Indirect connections (e.g., wrist-shoulder)
       - Expands receptive field for distant joint relations
    
    3. Channel-wise Attention (通道注意力)
       - A_adapt = softmax(φ(X) @ ψ(X)^T / √d)
       - Query-Key mechanism similar to Transformer
       - Dynamically generates adjacency per sample
    
    4. Gated Stability (门控稳定机制)
       - Sigmoid gating prevents gradient explosion
       - Improves training stability
    
    5. Multi-Head Design (多头设计)
       - Parallel learning of multiple relation patterns
       - Enhanced expressiveness
    
    Args:
        in_channels: Input channel dimension
        out_channels: Output channel dimension
        A: Adjacency matrix [num_head, V, V]
        num_heads: Number of attention heads (default: from A.shape[0])
        use_gate: Whether to use gating mechanism
        use_multiscale: Whether to use 2-hop adjacency
    """
    def __init__(self, in_channels, out_channels, A, num_heads=None, use_gate=True, use_multiscale=True):
        super(HDGC, self).__init__()
        self.in_c = in_channels
        self.out_c = out_channels
        self.num_head = A.shape[0] if num_heads is None else num_heads
        self.use_gate = use_gate
        self.use_multiscale = use_multiscale
        
        # Prior topology (learnable, initialized from skeleton anatomy)
        self.A_prior = nn.Parameter(torch.from_numpy(A.astype(np.float32)), requires_grad=True)
        
        # Multi-scale: compute 2-hop adjacency from 1-hop
        if use_multiscale:
            A_2hop = np.zeros_like(A)
            for h in range(A.shape[0]):
                A_h = A[h]
                A_2hop[h] = np.minimum(A_h @ A_h, 1.0)  # 2-hop connections
                A_2hop[h] = A_2hop[h] - A_h  # Remove 1-hop from 2-hop
                A_2hop[h] = np.clip(A_2hop[h], 0, 1)
            self.A_2hop = nn.Parameter(torch.from_numpy(A_2hop.astype(np.float32)), requires_grad=True)
            self.beta = nn.Parameter(torch.tensor(0.3))  # 2-hop scale factor
        
        # Adaptive topology: channel-wise attention (Query-Key)
        rel_channels = max(in_channels // 8, 8)
        self.phi = nn.Conv2d(in_channels, rel_channels * self.num_head, 1)  # Query
        self.psi = nn.Conv2d(in_channels, rel_channels * self.num_head, 1)  # Key
        self.scale = rel_channels ** -0.5
        
        # Learnable fusion weight λ
        self.lambda_adapt = nn.Parameter(torch.tensor(0.5))
        
        # Per-head convolutions
        self.conv_d = nn.ModuleList()
        for i in range(self.num_head):
            self.conv_d.append(nn.Conv2d(in_channels, out_channels, 1))
        
        # Residual connection
        if in_channels != out_channels:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.down = lambda x: x
        
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # Gating mechanism for stability
        if use_gate:
            self.gate = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.Sigmoid()
            )
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        bn_init(self.bn, 1e-6)
        for i in range(self.num_head):
            conv_branch_init(self.conv_d[i], self.num_head)
    
    def forward(self, x, attn=None):
        N, C, T, V = x.size()
        
        # Compute adaptive adjacency via attention
        phi = self.phi(x)  # Query
        psi = self.psi(x)  # Key
        
        phi = rearrange(phi, 'n (h d) t v -> (n t) h v d', h=self.num_head)
        psi = rearrange(psi, 'n (h d) t v -> (n t) h v d', h=self.num_head)
        
        # A_adapt = softmax(Q @ K^T / sqrt(d))
        A_adapt = torch.matmul(phi, psi.transpose(-1, -2)) * self.scale
        A_adapt = F.softmax(A_adapt, dim=-1)
        
        # Prior topology
        A_prior = self.A_prior.unsqueeze(0)
        
        # Multi-scale fusion: A_prior + β * A_2hop
        if self.use_multiscale:
            A_2hop = self.A_2hop.unsqueeze(0)
            A_prior_full = A_prior + self.beta * A_2hop
        else:
            A_prior_full = A_prior
        
        # Hybrid fusion: A_final = A_prior_full + λ * A_adapt
        lambda_val = torch.clamp(self.lambda_adapt, 0.0, 1.0)
        A_final = A_prior_full + lambda_val * A_adapt
        
        # Graph convolution per head
        out = None
        feature = rearrange(x, 'n c t v -> (n t) v c')
        for h in range(self.num_head):
            A_h = A_final[:, h, :, :]
            z = torch.bmm(A_h, feature)
            z = rearrange(z, '(n t) v c -> n c t v', t=T).contiguous()
            z = self.conv_d[h](z)
            out = z + out if out is not None else z
        
        out = self.bn(out)
        
        # Gating for stability
        if self.use_gate:
            gate = self.gate(x)
            out = gate * out
        
        # Residual connection
        out = out + self.down(x)
        out = self.relu(out)
        
        return out


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    """
    Temporal Attention with Progressive Attention Bias (PAB).
    Uses HDGC for Q/K/V projection.
    """
    def __init__(self, dim, seq_len, heads=8, dim_head=64, dropout=0.,
                 use_mask=False, A=1, num_point=25):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5
        self.use_mask = use_mask
        self.num_point = num_point
        self.seq_len = seq_len

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        # Use HDGC for Q/K/V projection
        self.to_qkv = HDGC(dim, inner_dim * 3, A, use_gate=True, use_multiscale=True)

        # PAB: Progressive Temporal Mask (causal)
        self.register_buffer("temporal_mask", torch.ones(seq_len, seq_len).tril()
                                     .view(1, 1, seq_len, seq_len))
        
        # PAB: Learnable relative position bias
        self.rel_pos_bias = nn.Parameter(torch.zeros(heads, seq_len))
        with torch.no_grad():
            for i in range(seq_len):
                self.rel_pos_bias.data[:, i] = -0.05 * i
        
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()
    
    def _get_rel_pos_bias(self, T):
        positions = torch.arange(T, device=self.rel_pos_bias.device)
        rel_pos = positions.unsqueeze(1) - positions.unsqueeze(0)
        rel_pos = rel_pos.clamp(min=0, max=self.seq_len - 1)
        bias = self.rel_pos_bias[:, rel_pos]
        return bias.unsqueeze(0)

    def forward(self, x):
        B, T, C = x.shape
        V = self.num_point

        x = rearrange(x, '(b v) t c -> b c t v', v=V)
        qkv = self.to_qkv(x)
        qkv = rearrange(qkv, 'b c t v -> (b v) t c', v=V)
        qkv = qkv.chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        attn = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = attn + self._get_rel_pos_bias(T)

        if self.use_mask:
            attn = attn.masked_fill(self.temporal_mask[:, :, :T, :T] == 0, float("-inf"))
        
        attn = self.attend(attn)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out), attn


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, max_seq_len,
                 dropout=0., use_mask=True, A=1, num_point=25):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(
                    dim, max_seq_len, heads=heads, dim_head=dim_head,
                    dropout=dropout, use_mask=use_mask, A=A, num_point=num_point)
                ),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))
        self._attns = []

    def forward(self, x):
        attns = []
        for attn, ff in self.layers:
            res = x
            x, sa = attn(x)
            x += res
            x = ff(x) + x
            attns.append(sa.clone())
        self._attns = attns
        return x

    def get_attns(self):
        return self._attns


def PositionalEncoding(d_model: int, dropout: float = 0.1, max_len: int = 5000):
    position = torch.arange(max_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
    pe = torch.zeros(1, max_len, d_model)
    pe[0, :, 0::2] = torch.sin(position * div_term)
    pe[0, :, 1::2] = torch.cos(position * div_term)
    return pe


class TemporalEncoder(nn.Module):
    """Temporal Encoder using Transformer with HDGC-based attention."""
    def __init__(self, seq_len, dim, depth, heads, mlp_dim, dim_head=64, 
                 dropout=0., emb_dropout=0., A=1, num_point=25, device='cuda'):
        super().__init__()

        self.pe = PositionalEncoding(d_model=dim, max_len=seq_len).to(device)
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(
            dim, depth, heads, dim_head, mlp_dim, seq_len, 
            dropout, A=A, num_point=num_point, use_mask=True
        )
        self.to_latent = nn.Identity()
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, x):
        B, C, T, V = x.shape
        x = rearrange(x, 'b c t v -> (b v) t c')
        x = x + self.pe[:, :T, :]
        x = self.transformer(x)
        x = self.to_latent(x)
        x = rearrange(x, '(b v) t c -> b c t v', v=V)
        return x

    def get_attention(self):
        return self.transformer._attns
