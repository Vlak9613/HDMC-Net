"""
HDMC-Net NTU RGB+D Skeleton Graph Definition

Defines the skeleton topology for NTU RGB+D dataset (25 joints).
"""

import sys
import numpy as np

sys.path.extend(['../'])
from graph import graph_utils

# NTU RGB+D skeleton has 25 joints
num_node = 25
self_link = [(i, i) for i in range(num_node)]

# Skeleton connectivity (parent-child relationships)
inward_ori_index = [
    (2, 1), (2, 21), (21, 3), (3, 4),  # head
    (21, 5), (5, 6), (6, 7), (7, 8), (8, 23), (23, 22),  # left arm
    (21, 9), (9, 10), (10, 11), (11, 12), (12, 25), (25, 24),  # right arm
    (1, 13), (13, 14), (14, 15), (15, 16),  # left leg
    (1, 17), (17, 18), (18, 19), (19, 20)  # right leg
]

inward = [(i - 1, j - 1) for (i, j) in inward_ori_index]
outward = [(j, i) for (i, j) in inward]
neighbor = inward + outward

# Coarse skeleton (11 joints)
num_node_1 = 11
indices_1 = [0, 3, 5, 7, 9, 11, 13, 15, 17, 19, 20]
self_link_1 = [(i, i) for i in range(num_node_1)]
inward_ori_index_1 = [(1, 11), (2, 11), (3, 11), (4, 3), (5, 11), (6, 5), (7, 1), (8, 7), (9, 1), (10, 9)]
inward_1 = [(i - 1, j - 1) for (i, j) in inward_ori_index_1]
outward_1 = [(j, i) for (i, j) in inward_1]
neighbor_1 = inward_1 + outward_1

# Super coarse skeleton (5 joints)
num_node_2 = 5
indices_2 = [3, 5, 6, 8, 10]
self_link_2 = [(i, i) for i in range(num_node_2)]
inward_ori_index_2 = [(0, 4), (1, 4), (2, 4), (3, 4), (0, 1), (2, 3)]
inward_2 = [(i - 1, j - 1) for (i, j) in inward_ori_index_2]
outward_2 = [(j, i) for (i, j) in inward_2]
neighbor_2 = inward_2 + outward_2


class Graph:
    """NTU RGB+D Skeleton Graph."""
    
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
        self.A_outward_binary = graph_utils.get_adjacency_matrix(self.outward, self.num_node)
        self.A = self.get_adjacency_matrix(labeling_mode)
        self.A1 = graph_utils.get_spatial_graph(num_node_1, self_link_1, inward_1, outward_1)
        self.A2 = graph_utils.get_spatial_graph(num_node_2, self_link_2, inward_2, outward_2)
        self.A_binary = graph_utils.edge2mat(neighbor, num_node)
        self.A_norm = graph_utils.normalize_adjacency_matrix(self.A_binary + 2*np.eye(num_node))
        self.A_binary_K = graph_utils.get_k_scale_graph(scale, self.A_binary)

        # Hierarchical pooling matrices
        self.A_A1 = ((self.A_binary + np.eye(num_node)) / np.sum(self.A_binary + np.eye(self.A_binary.shape[0]), axis=1, keepdims=True))[indices_1]
        self.A1_A2 = graph_utils.edge2mat(neighbor_1, num_node_1) + np.eye(num_node_1)
        self.A1_A2 = (self.A1_A2 / np.sum(self.A1_A2, axis=1, keepdims=True))[indices_2]

    def get_adjacency_matrix(self, labeling_mode=None):
        """Get adjacency matrix based on labeling mode."""
        if labeling_mode is None:
            return self.A
        if labeling_mode == 'spatial':
            A = graph_utils.get_spatial_graph(num_node, self_link, inward, outward)
        else:
            raise ValueError()
        return A
