"""
HDMC-Net Loss Functions

Loss functions for training HDMC-Net model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def masked_recon_loss(x, x_hat, mask):
    """
    Masked reconstruction loss.
    
    Computes MSE loss only on valid (non-masked) regions.
    
    Args:
        x: Ground truth skeleton [B, C, T, V, M]
        x_hat: Predicted skeleton [B, C, T, V, M]
        mask: Binary mask indicating valid regions
        
    Returns:
        Scalar loss value
    """
    recon_loss = (F.mse_loss(x_hat, x, reduction="none") * mask).sum() / mask.sum()
    return recon_loss


class LabelSmoothingCrossEntropy(nn.Module):
    """
    Label smoothing cross entropy loss.
    
    Implements label smoothing regularization to prevent overconfidence
    and improve generalization.
    
    Args:
        smoothing: Smoothing factor (default: 0.1)
        T: Temporal dimension (unused, for API compatibility)
    """
    def __init__(self, smoothing=0.1, T=None):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing

    def forward(self, x, target):
        """
        Compute label smoothing cross entropy loss.
        
        Args:
            x: Logits [B, num_classes]
            target: Ground truth labels [B]
            
        Returns:
            Scalar loss value
        """
        confidence = 1. - self.smoothing
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()
