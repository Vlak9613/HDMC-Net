"""
HDMC-Net NW-UCLA Skeleton Graph Definition

Defines the skeleton topology for NW-UCLA dataset (20 joints).
"""

import sys
import numpy as np

sys.path.extend(['../'])
from graph import graph_utils

# NW-UCLA skeleton has 20 joints
num_node = 20
self_link = [(i, i) for i in range(num_node)]

# Skeleton connectivity
inward_ori_index = [
    (1, 2), (2, 3), (4, 3), (5, 3), (6, 5), (7, 6),
    (8, 7), (9, 3), (10, 9), (11, 10), (12, 11), (13, 1),
    (14, 13), (15, 14), (16, 15), (17, 1), (18, 17), (19, 18),
    (20, 19)
]

inward = [(i - 1, j - 1) for (i, j) in inward_ori_index]
outward = [(j, i) for (i, j) in inward]
neighbor = inward + outward


class Graph:
    """NW-UCLA Skeleton Graph."""
    
    def __init__(self, labeling_mode='spatial', scale=1):
        """
        Args:
            labeling_mode: Graph labeling strategy ('spatial')
            scale: k-hop scale for graph
        """
        self.num_node = num_node
        self.self_link = self_link
        self.inward = inward
        self.outward = outward
        self.neighbor = neighbor
        
        # Adjacency matrices
        self.A = self.get_adjacency_matrix(labeling_mode)
        self.A_outward_binary = graph_utils.get_adjacency_matrix(self.outward, self.num_node)
        self.A_binary = graph_utils.edge2mat(neighbor, num_node)
        self.A_norm = graph_utils.normalize_adjacency_matrix(self.A_binary + 2*np.eye(num_node))
        self.A_binary_K = graph_utils.get_k_scale_graph(scale, self.A_binary)

    def get_adjacency_matrix(self, labeling_mode=None):
        """Get adjacency matrix based on labeling mode."""
        if labeling_mode is None:
            return self.A
        if labeling_mode == 'spatial':
            A = graph_utils.get_spatial_graph(num_node, self_link, inward, outward)
        else:
            raise ValueError()
        return A
