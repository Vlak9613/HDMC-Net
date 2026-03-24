"""
HDMC-Net Model Utilities

Utility functions for model initialization and operations.
"""

import torch
import math
import torch.nn as nn


def import_class(name):
    """Dynamically import a class from a module path string."""
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


def bn_init(bn, scale):
    """Initialize batch normalization."""
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)


def conv_branch_init(conv, branches):
    """Initialize convolution branch."""
    weight = conv.weight
    n = weight.size(0)
    k1 = weight.size(1)
    k2 = weight.size(2)
    nn.init.normal_(weight, 0, math.sqrt(2. / (n * k1 * k2 * branches)))
    if conv.bias is not None:
        nn.init.constant_(conv.bias, 0)


def conv_init(conv):
    """Initialize convolution layer."""
    if conv.weight is not None:
        nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    if conv.bias is not None:
        nn.init.constant_(conv.bias, 0)


def sample_standard_gaussian(mu, sigma):
    """Sample from standard Gaussian distribution."""
    device = mu.device
    d = torch.distributions.normal.Normal(
        torch.Tensor([0.]).to(device), 
        torch.Tensor([1.]).to(device)
    )
    r = d.sample(mu.size()).squeeze(-1)
    return r * sigma.float() + mu.float()


def cum_mean_pooling(x, denom, dim=-1):
    """Cumulative mean pooling."""
    x = torch.cumsum(x, dim=dim) / denom
    return x


def cum_max_pooling(x, denom, dim=-1):
    """Cumulative max pooling."""
    x, idx = torch.cummax(x, dim=dim)
    return x


def identity(x, denom, dim=-1):
    """Identity function (no pooling)."""
    return x


def max_pooling(x, dim=-1):
    """Max pooling along dimension."""
    x, idx = torch.max(x, dim=dim)
    return x
