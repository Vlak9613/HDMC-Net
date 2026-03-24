"""
MomentumNet Extrapolator - Physics-guided extrapolator for online skeleton action recognition.

Core Design:
============
Use velocity from data to guarantee non-zero increments:
    v = z[:, :, -1] - z[:, :, -2]  (velocity from data, guaranteed non-zero)
    Δv = Network(z, v)             (network learns velocity correction)
    z_hat = z_last + (v + Δv) * dt (even if Δv=0, base velocity v exists)

Key Features:
=============
1. Physics-guided: Velocity-based prediction mimics physical motion
2. Non-zero guarantee: Base velocity from data ensures meaningful predictions
3. Learnable dynamics: Network learns correction to base velocity
4. HDGC integration: Uses Hybrid Dynamic Graph Convolution for spatial modeling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange

from model.layers import HDGC


class SimpleGraphConv(nn.Module):
    """Simple fixed graph convolution for lightweight operations."""
    def __init__(self, in_channels, out_channels, A):
        super().__init__()
        if isinstance(A, np.ndarray):
            if A.ndim == 3:
                A = A.mean(axis=0)
            A = torch.from_numpy(A).float()
        self.register_buffer('A', A)
        self.conv = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        squeeze_flag = False
        if x.dim() == 3:
            x = x.unsqueeze(2)
            squeeze_flag = True
        
        B, C, T, V = x.size()
        x = rearrange(x, 'b c t v -> (b t) v c')
        x = torch.matmul(self.A.to(x.dtype), x)
        x = rearrange(x, '(b t) v c -> b c t v', b=B)
        x = self.conv(x)
        x = self.bn(x)
        
        if squeeze_flag:
            x = x.squeeze(2)
        return x


class VelocityEncoder(nn.Module):
    """Encodes velocity information using simple graph convolutions."""
    def __init__(self, dim, A):
        super().__init__()
        self.encoder = nn.Sequential(
            SimpleGraphConv(dim, dim, A),
            nn.LeakyReLU(0.1, inplace=False),
            SimpleGraphConv(dim, dim, A),
        )
        
    def forward(self, velocity):
        return self.encoder(velocity)


class VelocityCorrector(nn.Module):
    """
    Learns velocity correction using HDGC.
    
    Combines current state and encoded velocity to predict Δv.
    """
    def __init__(self, dim, A, num_head=3):
        super().__init__()
        
        if isinstance(A, np.ndarray):
            if A.ndim == 2:
                A = np.stack([A] * num_head, axis=0)
        
        # Feature fusion with HDGC (captures dynamic joint relations)
        self.fusion = HDGC(dim * 2, dim, A, use_gate=True, use_multiscale=True)
        
        # Correction prediction
        self.corrector = nn.Sequential(
            nn.Conv1d(dim, dim, 1, bias=False),
            nn.LeakyReLU(0.1, inplace=False),
            nn.Conv1d(dim, dim, 1, bias=False),
        )
        
        self.correction_scale = nn.Parameter(torch.ones(1) * 0.5)
        
    def forward(self, z_last, v_encoded):
        combined = torch.cat([z_last, v_encoded], dim=1)
        combined = combined.unsqueeze(2)
        
        fused = self.fusion(combined).squeeze(2)
        delta_v = self.corrector(fused)
        delta_v = self.correction_scale * delta_v
        
        return delta_v


class MomentumNetExtrapolator(nn.Module):
    """
    MomentumNet Extrapolator - Physics-guided future prediction.
    
    Core Formula:
    =============
    v = z[:, :, -1] - z[:, :, -2]      # Current velocity (from data)
    v_encoded = VelocityEncoder(v)     # Encode velocity
    Δv = VelocityCorrector(z, v_enc)   # Learn correction
    v_new = decay * v + Δv             # Update velocity
    z_next = z_last + v_new * dt       # Extrapolate next state
    
    Advantages:
    ===========
    1. Non-zero guarantee: Base velocity v is computed from data, always non-zero
    2. Physics intuition: Mimics physical motion with velocity and acceleration
    3. Learnable dynamics: Network learns to correct velocity based on context
    4. HDGC integration: Dynamic graph topology for joint relation modeling
    """
    
    def __init__(self, dim, A, n_step=3, T=64, V=25, dropout=0.1, num_head=3):
        super().__init__()
        self.dim = dim
        self.n_step = n_step
        self.T = T
        self.V = V
        
        if isinstance(A, np.ndarray):
            if A.ndim == 2:
                A_multi = np.stack([A] * num_head, axis=0)
            else:
                A_multi = A
        else:
            A_multi = A
        self.A = A
        
        # Velocity encoder (lightweight)
        self.velocity_encoder = VelocityEncoder(dim, A)
        
        # Velocity corrector (with HDGC)
        self.velocity_corrector = VelocityCorrector(dim, A_multi, num_head)
        
        # Input projection with HDGC
        self.input_proj = nn.Sequential(
            HDGC(dim, dim, A_multi, use_gate=True, use_multiscale=True),
            nn.LeakyReLU(0.1, inplace=False),
            nn.Dropout(dropout)
        )
        
        # Output projection with HDGC
        self.output_proj = nn.Sequential(
            HDGC(dim, dim, A_multi, use_gate=True, use_multiscale=True),
            nn.LeakyReLU(0.1, inplace=False),
        )
        
        # Learnable parameters
        if n_step > 0:
            self.step_embed = nn.Parameter(torch.randn(n_step, dim) * 0.01)
        self.dt = nn.Parameter(torch.ones(1) * 1.0)  # Time step
        self.velocity_decay = nn.Parameter(torch.ones(1) * 0.95)  # Friction
        
        self._print_info()
        
    def _print_info(self):
        params = sum(p.numel() for p in self.parameters())
        print(f"[MomentumNet] Parameters: {params:,}, n_step: {self.n_step}")
        
    def compute_velocity(self, z):
        """Compute velocity from feature sequence (first-order difference)."""
        velocity = z[:, :, -1, :] - z[:, :, -2, :]
        return velocity
        
    def forward(self, z_0, t=None):
        """
        Forward pass for future extrapolation.
        
        Args:
            z_0: [B*T, C, V] - Flattened encoder output
            t: Time steps (ignored, for API compatibility)
            
        Returns:
            zs: [n_step+1, B*T, C, V] - Initial state + n_step predictions
        """
        if self.n_step == 0:
            return z_0.unsqueeze(0)  # [1, BT, C, V]

        BT, C, V = z_0.size()
        T = self.T
        B = BT // T
        
        z_encoder = rearrange(z_0, '(b t) c v -> b c t v', t=T)
        
        # Compute initial velocity from data
        velocity = self.compute_velocity(z_encoder)
        v_encoded = self.velocity_encoder(velocity)
        
        # Autoregressive extrapolation
        all_predictions = []
        z_current = z_encoder.clone()
        current_velocity = velocity.clone()
        
        for step in range(self.n_step):
            # Get last frame
            z_last = z_current[:, :, -1, :]
            
            # Input projection with HDGC
            z_proj = self.input_proj(z_last.unsqueeze(2)).squeeze(2)
            
            # Add step embedding
            step_emb = self.step_embed[step].view(1, C, 1)
            z_proj = z_proj + step_emb
            
            # Compute velocity correction
            delta_v = self.velocity_corrector(z_proj, v_encoded)
            
            # Update velocity with decay (simulates friction)
            decay = torch.sigmoid(self.velocity_decay)
            updated_velocity = decay * current_velocity + delta_v
            
            # Extrapolate next frame: z_next = z_last + v * dt
            dt = torch.abs(self.dt)
            z_next_frame = z_last + updated_velocity * dt
            
            # Output projection with HDGC
            z_next_frame = self.output_proj(z_next_frame.unsqueeze(2)).squeeze(2)
            
            # Build next sequence (shift left, add new frame)
            z_next = torch.cat([
                z_current[:, :, 1:, :],
                z_next_frame.unsqueeze(2)
            ], dim=2)
            
            all_predictions.append(z_next)
            z_current = z_next
            current_velocity = updated_velocity
            
            # Update velocity encoding for next step
            if step < self.n_step - 1:
                new_velocity = self.compute_velocity(z_current)
                v_encoded = self.velocity_encoder(new_velocity)
        
        # Stack predictions
        z_hat = torch.stack(all_predictions, dim=0)
        z_hat = rearrange(z_hat, 'n b c t v -> n (b t) c v')
        
        # Prepend initial state
        zs = torch.cat([z_0.unsqueeze(0), z_hat], dim=0)
        
        return zs
