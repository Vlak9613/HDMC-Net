"""
HDMC-Net UCLA Dataset Feeder

Data loader for NW-UCLA skeleton action recognition dataset.
"""

import numpy as np
import pickle
import json
import random
import math
import os
import glob

from torch.utils.data import Dataset


class Feeder(Dataset):
    """NW-UCLA Dataset Feeder."""
    
    def __init__(self, data_path, split=None, p_interval=None, repeat=5, random_choose=False, 
                 random_shift=False, random_move=False, random_rot=False, window_size=52, 
                 normalization=False, debug=False, use_mmap=True, bone=False, vel=False, 
                 sort=False, A=None):
        """
        Args:
            data_path: Path to data directory
            split: 'train' or 'test'
            p_interval: Observation rate interval
            repeat: Number of times to repeat data
            random_choose: Randomly choose sequence portion
            random_shift: Random temporal shift
            random_move: Random spatial transformation
            random_rot: Random rotation
            window_size: Output sequence length
            normalization: Normalize input
            debug: Debug mode
            use_mmap: Use memory mapping
            bone: Use bone features
            vel: Use velocity features
            sort: Sort by label
            A: Adjacency matrix
        """
        self.nw_ucla_root = 'data/NW-UCLA/all_sqe/'
        
        # UCLA dataset split: train = view 1,2; test = view 3
        if 'test' in split:
            self.train_val = 'test'
            self.data_dict = self._get_test_data_dict()
        else:
            self.train_val = 'train'
            self.data_dict = self._get_train_data_dict()
            
        self.bone = [(1, 2), (2, 3), (3, 3), (4, 3), (5, 3), (6, 5), (7, 6), (8, 7), 
                     (9, 3), (10, 9), (11, 10), (12, 11), (13, 1), (14, 13), (15, 14), 
                     (16, 15), (17, 1), (18, 17), (19, 18), (20, 19)]
        self.label = []
        for index in range(len(self.data_dict)):
            info = self.data_dict[index]
            self.label.append(int(info['label']) - 1)

        self.debug = debug
        self.data_path = data_path
        self.label_path = split
        self.random_choose = random_choose
        self.random_shift = random_shift
        self.random_move = random_move
        self.window_size = window_size
        self.normalization = normalization
        self.use_mmap = use_mmap
        self.repeat = repeat if split == "train" else 1
        self.A = A
        self.load_data()
        if normalization:
            self.get_mean_map()

    def _scan_data_dict(self, views):
        """Scan JSON files and build data dictionary for given views.
        
        Args:
            views: List of view strings to include, e.g. ['v01', 'v02']
        
        Returns:
            List of dicts with 'file_name' and 'label' keys.
        """
        data_dict = []
        for json_path in sorted(glob.glob(os.path.join(self.nw_ucla_root, '*.json'))):
            fname = os.path.splitext(os.path.basename(json_path))[0]
            # filename format: a{action}_s{subject}_e{episode}_v{view}
            view = fname.split('_')[-1]  # e.g. 'v01'
            if view in views:
                with open(json_path, 'r') as f:
                    info = json.load(f)
                data_dict.append({
                    'file_name': fname,
                    'label': info['label'],
                })
        return data_dict

    def _get_test_data_dict(self):
        """Get test data dictionary (view 3)."""
        return self._scan_data_dict(['v03'])
        
    def _get_train_data_dict(self):
        """Get train data dictionary (views 1, 2)."""
        return self._scan_data_dict(['v01', 'v02'])

    def load_data(self):
        """Load skeleton data from JSON files."""
        self.data = []
        for data in self.data_dict:
            file_name = data['file_name']
            with open(self.nw_ucla_root + file_name + '.json', 'r') as f:
                json_file = json.load(f)
            skeletons = json_file['skeletons']
            value = np.array(skeletons)
            self.data.append(value)

    def get_mean_map(self):
        """Compute mean and std for normalization."""
        data = self.data
        N, C, T, V, M = data.shape
        self.mean_map = data.mean(axis=2, keepdims=True).mean(axis=4, keepdims=True).mean(axis=0)
        self.std_map = data.transpose((0, 2, 4, 1, 3)).reshape((N * T * M, C * V)).std(axis=0).reshape((C, 1, V, 1))

    def __len__(self):
        return len(self.data_dict) * self.repeat

    def __iter__(self):
        return self

    def rand_view_transform(self, X, agx, agy, s):
        """Random view transformation for data augmentation."""
        agx = math.radians(agx)
        agy = math.radians(agy)
        Rx = np.asarray([[1, 0, 0], [0, math.cos(agx), math.sin(agx)], [0, -math.sin(agx), math.cos(agx)]])
        Ry = np.asarray([[math.cos(agy), 0, -math.sin(agy)], [0, 1, 0], [math.sin(agy), 0, math.cos(agy)]])
        Ss = np.asarray([[s, 0, 0], [0, s, 0], [0, 0, s]])
        X0 = np.dot(np.reshape(X, (-1, 3)), np.dot(Ry, np.dot(Rx, Ss)))
        X = np.reshape(X0, X.shape)
        return X

    def __getitem__(self, index):
        label = self.label[index % len(self.data_dict)]
        value = self.data[index % len(self.data_dict)]

        if self.train_val == 'train':
            random.random()
            agx = random.randint(-60, 60)
            agy = random.randint(-60, 60)
            s = random.uniform(0.5, 1.5)

            center = value[0, 1, :]
            value = value - center
            scalerValue = self.rand_view_transform(value, agx, agy, s)

            scalerValue = np.reshape(scalerValue, (-1, 3))
            scalerValue = (scalerValue - np.min(scalerValue, axis=0)) / (np.max(scalerValue, axis=0) - np.min(scalerValue, axis=0))
            scalerValue = scalerValue * 2 - 1
            scalerValue = np.reshape(scalerValue, (-1, 20, 3))

            data = np.zeros((self.window_size, 20, 3))

            value = scalerValue[:, :, :]
            length = value.shape[0]

            random_idx = random.sample(list(np.arange(length)) * 100, self.window_size)
            random_idx.sort()
            data[:, :, :] = value[random_idx, :, :]

        else:
            random.random()
            agx = 0
            agy = 0
            s = 1.0

            center = value[0, 1, :]
            value = value - center
            scalerValue = self.rand_view_transform(value, agx, agy, s)

            scalerValue = np.reshape(scalerValue, (-1, 3))
            scalerValue = (scalerValue - np.min(scalerValue, axis=0)) / (np.max(scalerValue, axis=0) - np.min(scalerValue, axis=0))
            scalerValue = scalerValue * 2 - 1

            scalerValue = np.reshape(scalerValue, (-1, 20, 3))

            data = np.zeros((self.window_size, 20, 3))

            value = scalerValue[:, :, :]
            length = value.shape[0]

            idx = np.linspace(0, length-1, self.window_size).astype(int)
            data[:, :, :] = value[idx, :, :]  # T,V,C

        if 'bone' in self.data_path:
            data_bone = np.zeros_like(data)
            for bone_idx in range(20):
                data_bone[:, self.bone[bone_idx][0] - 1, :] = data[:, self.bone[bone_idx][0] - 1, :] - data[:, self.bone[bone_idx][1] - 1, :]
            data = data_bone

        if 'motion' in self.data_path:
            data_motion = np.zeros_like(data)
            data_motion[:-1, :, :] = data[1:, :, :] - data[:-1, :, :]
            data = data_motion
            
        data = np.transpose(data, (2, 0, 1))
        C, T, V = data.shape
        data = np.reshape(data, (C, T, V, 1))
        mask = np.ones_like(data)

        return data, label, mask, index

    def top_k(self, score, top_k):
        """Compute top-k accuracy."""
        rank = score.argsort()
        hit_top_k = [l in rank[i, -top_k:] for i, l in enumerate(self.label)]
        return sum(hit_top_k) * 1.0 / len(hit_top_k)


def import_class(name):
    """Dynamically import a class from module path."""
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod
