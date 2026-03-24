"""
HDMC-Net HRC-Assembly Dataset Feeder

Data loader for HRC-Assembly skeleton action recognition dataset.
特点:
  - 单人数据 (M=1, 25 关节, Azure Kinect)
  - repeat 机制（模仿 UCLA）增加小数据集的训练样本量
  - Azure Kinect mm → meters 坐标缩放
  - 修正的 valid_frame_num 计算（适配 M=1）
  - 保守旋转增强 (theta=0.3)，适合微动作
"""

import numpy as np
import pickle

from torch.utils.data import Dataset

from feeders import feeder_utils


class Feeder(Dataset):
    """HRC-Assembly Dataset Feeder."""

    def __init__(self, data_path, label_path=None, p_interval=1, split='train', repeat=5,
                 random_choose=False, random_shift=False, random_move=False, random_rot=False,
                 window_size=64, normalization=False, debug=False, use_mmap=False,
                 vel=False, sort=False, A=None):
        """
        Args:
            data_path: Path to data file (.npz)
            label_path: Path to label file (optional, unused)
            p_interval: Observation rate interval [min, max] or single value
            split: 'train' or 'test'
            repeat: Number of times to repeat data per epoch (like UCLA)
            random_choose: Randomly choose a portion of sequence
            random_shift: Randomly shift sequence temporally
            random_move: Random spatial transformation
            random_rot: Random rotation augmentation
            window_size: Output sequence length
            normalization: Normalize input
            debug: Use only first 100 samples
            use_mmap: Use memory mapping for data loading
            vel: Use velocity features
            sort: Sort data by label
            A: Adjacency matrix for graph transformation
        """
        self.debug = debug
        self.data_path = data_path
        self.label_path = label_path
        self.split = split
        self.random_choose = random_choose
        self.random_shift = random_shift
        self.random_move = random_move
        self.window_size = window_size
        self.normalization = normalization
        self.use_mmap = use_mmap
        self.p_interval = p_interval
        self.random_rot = random_rot
        self.vel = vel
        self.A = A
        # repeat 机制: 训练时重复 repeat 次, 测试时不重复
        self.repeat = repeat if split == 'train' else 1
        self.load_data()
        if sort:
            self.get_n_per_class()
            self.sort()
        if normalization:
            self.get_mean_map()

    def load_data(self):
        """Load data from npz file.
        
        HRC 数据格式: (N, T, 150) 其中 150 = 2人×25关节×3坐标
        但 HRC 是单人数据，后 75 维为零填充。
        """
        npz_data = np.load(self.data_path)
        if self.split == 'train':
            self.data = npz_data['x_train']
            self.label = np.argmax(npz_data['y_train'], axis=-1)
        elif self.split == 'test':
            self.data = npz_data['x_test']
            self.label = np.argmax(npz_data['y_test'], axis=-1)
        else:
            raise NotImplementedError('data split only supports train/test')

        # 过滤 NaN 样本
        nan_out = np.isnan(self.data.mean(-1).mean(-1)) == False
        self.data = self.data[nan_out]
        self.label = self.label[nan_out]

        self.sample_name = [self.split + '_' + str(i) for i in range(len(self.data))]

        N, T, _ = self.data.shape

        # 单人优化: 只取前 75 维 (第一个人)，避免浪费计算在零填充的第二人上
        self.data = self.data[:, :, :75]

        # 骨架对齐变换 (如果有)
        if self.A is not None:
            self.data = self.data.reshape((N * T, 25, 3))
            self.data = np.array(self.A) @ self.data

        # reshape 为 (N, C, T, V, M) 格式, M=1 (单人)
        self.data = self.data.reshape(N, T, 1, 25, 3).transpose(0, 4, 1, 3, 2)

        # Azure Kinect mm -> meters, 与模型参数尺度对齐
        self.data = self.data / 1000.0

    def get_n_per_class(self):
        """Get number of samples per class."""
        self.n_per_cls = np.zeros(len(self.label), dtype=int)
        for label in self.label:
            self.n_per_cls[label] += 1
        self.csum_n_per_cls = np.insert(np.cumsum(self.n_per_cls), 0, 0)

    def sort(self):
        """Sort data by label."""
        sorted_idx = self.label.argsort()
        self.data = self.data[sorted_idx]
        self.label = self.label[sorted_idx]

    def get_mean_map(self):
        """Compute mean and std for normalization."""
        data = self.data
        N, C, T, V, M = data.shape
        self.mean_map = data.mean(axis=2, keepdims=True).mean(axis=4, keepdims=True).mean(axis=0)
        self.std_map = data.transpose((0, 2, 4, 1, 3)).reshape((N * T * M, C * V)).std(axis=0).reshape((C, 1, V, 1))

    def __len__(self):
        """repeat 机制: 训练时返回 repeat 倍的长度"""
        return len(self.label) * self.repeat

    def __iter__(self):
        return self

    def __getitem__(self, index):
        # repeat 机制: 取模获得真实索引
        real_index = index % len(self.label)
        data_numpy = np.array(self.data[real_index])
        label = self.label[real_index]

        # 计算有效帧数 —— 修正版 (适配 M=1)
        # data_numpy shape: (C=3, T, V=25, M=1)
        valid_frame = data_numpy.sum(0, keepdims=True).sum(2, keepdims=True)  # (1, T, 1, 1)
        # 对 M=1 的数据，显式 squeeze 掉 axis 0, 2, 3，只留 T 维
        valid_frame_num = np.sum(np.squeeze(valid_frame, axis=(0, 2, 3)) != 0)

        # Temporal crop + resize
        data_numpy = feeder_utils.valid_crop_resize(
            data_numpy, valid_frame_num, self.p_interval, self.window_size)
        mask = (abs(data_numpy.sum(0, keepdims=True).sum(2, keepdims=True)) > 0)

        # 保守旋转增强 (theta=0.3)，适合微动作
        if self.random_rot:
            data_numpy = feeder_utils.random_rot(data_numpy, theta=0.3)

        # 速度特征
        if self.vel:
            data_numpy[:, :-1] = data_numpy[:, 1:] - data_numpy[:, :-1]
            data_numpy[:, -1] = 0

        return data_numpy, label, mask, index

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
