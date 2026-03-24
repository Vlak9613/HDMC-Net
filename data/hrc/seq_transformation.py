# HRC-Assembly Dataset — Sequence Transformation Script
# Adapted from NTU seq_transformation.py for the HRC-Assembly dataset.
#
# Dataset parameters:
#   - 17 action classes (A01-A17)
#   - 9 subjects (P01-P09)
#   - 10 repetitions (R01-R10)
#   - Single person, 25 joints, Azure Kinect
#
# Cross-Subject split:
#   - Train: P01-P06 (1020 samples)
#   - Test:  P07-P09 (510 samples)

import sys
sys.path.append('../..')
import os
import os.path as osp
import numpy as np
import pickle
import logging
from utils import create_aligned_dataset

# ============================================================
#  HRC-Assembly Dataset Parameters
# ============================================================
NUM_CLASSES = 17          # Number of action classes (A01-A17)
NUM_PERFORMERS = 9        # Total number of subjects (P01-P09)

# Cross-Subject split
TRAIN_SUBJECT_IDS = [1, 2, 3, 4, 5, 6]
TEST_SUBJECT_IDS  = [7, 8, 9]

# ============================================================
#  Path Configuration
# ============================================================
root_path = './'
stat_path = osp.join(root_path, 'statistics')
camera_file = osp.join(stat_path, 'camera.txt')
performer_file = osp.join(stat_path, 'performer.txt')
replication_file = osp.join(stat_path, 'replication.txt')
label_file = osp.join(stat_path, 'label.txt')
skes_name_file = osp.join(stat_path, 'skes_available_name.txt')

denoised_path = osp.join(root_path, 'denoised_data')
raw_skes_joints_pkl = osp.join(denoised_path, 'raw_denoised_joints.pkl')
frames_file = osp.join(denoised_path, 'frames_cnt.txt')

save_path = './'

if not osp.exists(save_path):
    os.mkdir(save_path)


def remove_nan_frames(ske_name, ske_joints, nan_logger):
    num_frames = ske_joints.shape[0]
    valid_frames = []

    for f in range(num_frames):
        if not np.any(np.isnan(ske_joints[f])):
            valid_frames.append(f)
        else:
            nan_indices = np.where(np.isnan(ske_joints[f]))[0]
            nan_logger.info('{}\t{:^5}\t{}'.format(ske_name, f + 1, nan_indices))

    return ske_joints[valid_frames]


def seq_translation(skes_joints):
    """Sequence translation (centering): use joint-2 (mid-spine) of the first frame as origin."""
    for idx, ske_joints in enumerate(skes_joints):
        num_frames = ske_joints.shape[0]
        num_bodies = 1 if ske_joints.shape[1] == 75 else 2
        if num_bodies == 2:
            missing_frames_1 = np.where(ske_joints[:, :75].sum(axis=1) == 0)[0]
            missing_frames_2 = np.where(ske_joints[:, 75:].sum(axis=1) == 0)[0]
            cnt1 = len(missing_frames_1)
            cnt2 = len(missing_frames_2)

        i = 0  # get the "real" first frame of actor1
        while i < num_frames:
            if np.any(ske_joints[i, :75] != 0):
                break
            i += 1

        origin = np.copy(ske_joints[i, 3:6])  # new origin: joint-2

        for f in range(num_frames):
            if num_bodies == 1:
                ske_joints[f] -= np.tile(origin, 25)
            else:  # for 2 actors
                ske_joints[f] -= np.tile(origin, 50)

        if (num_bodies == 2) and (cnt1 > 0):
            ske_joints[missing_frames_1, :75] = np.zeros((cnt1, 75), dtype=np.float32)

        if (num_bodies == 2) and (cnt2 > 0):
            ske_joints[missing_frames_2, 75:] = np.zeros((cnt2, 75), dtype=np.float32)

        skes_joints[idx] = ske_joints  # Update

    return skes_joints


def frame_translation(skes_joints, skes_name, frames_cnt):
    nan_logger = logging.getLogger('nan_skes')
    nan_logger.setLevel(logging.INFO)
    nan_logger.addHandler(logging.FileHandler("./nan_frames.log"))
    nan_logger.info('{}\t{}\t{}'.format('Skeleton', 'Frame', 'Joints'))

    for idx, ske_joints in enumerate(skes_joints):
        num_frames = ske_joints.shape[0]
        # Calculate the distance between spine base (joint-1) and spine (joint-21)
        j1 = ske_joints[:, 0:3]
        j21 = ske_joints[:, 60:63]
        dist = np.sqrt(((j1 - j21) ** 2).sum(axis=1))

        for f in range(num_frames):
            origin = ske_joints[f, 3:6]  # new origin: middle of the spine (joint-2)
            if (ske_joints[f, 75:] == 0).all():
                ske_joints[f, :75] = (ske_joints[f, :75] - np.tile(origin, 25)) / \
                                      dist[f] + np.tile(origin, 25)
            else:
                ske_joints[f] = (ske_joints[f] - np.tile(origin, 50)) / \
                                 dist[f] + np.tile(origin, 50)

        ske_name = skes_name[idx]
        ske_joints = remove_nan_frames(ske_name, ske_joints, nan_logger)
        frames_cnt[idx] = num_frames  # update valid number of frames
        skes_joints[idx] = ske_joints

    return skes_joints, frames_cnt


def align_frames(skes_joints, frames_cnt):
    """
    Align all sequences to the same frame length.
    HRC data is single-person (75 dims), zero-padded to 150 dims for dual-person format compatibility.
    """
    num_skes = len(skes_joints)
    max_num_frames = frames_cnt.max()
    aligned_skes_joints = np.zeros((num_skes, max_num_frames, 150), dtype=np.float32)

    for idx, ske_joints in enumerate(skes_joints):
        num_frames = ske_joints.shape[0]
        num_bodies = 1 if ske_joints.shape[1] == 75 else 2
        if num_bodies == 1:
            aligned_skes_joints[idx, :num_frames] = np.hstack((ske_joints,
                                                               np.zeros_like(ske_joints)))
        else:
            aligned_skes_joints[idx, :num_frames] = ske_joints

    return aligned_skes_joints


def one_hot_vector(labels):
    num_skes = len(labels)
    labels_vector = np.zeros((num_skes, NUM_CLASSES))
    for idx, l in enumerate(labels):
        labels_vector[idx, l] = 1

    return labels_vector


def split_dataset(skes_joints, label, performer, evaluation, save_path):
    train_indices, test_indices = get_indices(performer, evaluation)

    # Save labels and num_frames for each sequence of each data set
    train_labels = label[train_indices]
    test_labels = label[test_indices]

    train_x = skes_joints[train_indices]
    train_y = one_hot_vector(train_labels)
    test_x = skes_joints[test_indices]
    test_y = one_hot_vector(test_labels)

    save_name = 'HRC_%s.npz' % evaluation
    np.savez(save_name, x_train=train_x, y_train=train_y, x_test=test_x, y_test=test_y)

    print('  Train: %d samples, Test: %d samples' % (len(train_x), len(test_x)))


def get_indices(performer, evaluation='CS'):
    test_indices = np.empty(0)
    train_indices = np.empty(0)

    if evaluation == 'CS':  # Cross Subject
        train_ids = TRAIN_SUBJECT_IDS
        test_ids = TEST_SUBJECT_IDS

        # Get indices of test data
        for idx in test_ids:
            temp = np.where(performer == idx)[0]  # 0-based index
            test_indices = np.hstack((test_indices, temp)).astype(int)

        # Get indices of training data
        for train_id in train_ids:
            temp = np.where(performer == train_id)[0]  # 0-based index
            train_indices = np.hstack((train_indices, temp)).astype(int)
    else:
        raise ValueError('HRC dataset only supports CS evaluation')

    return train_indices, test_indices


if __name__ == '__main__':
    print('=' * 60)
    print('  HRC-Assembly Sequence Transformation')
    print('=' * 60)

    print('[1/6] Loading metadata files...')
    camera = np.loadtxt(camera_file, dtype=int)    # camera id (all 1)
    performer = np.loadtxt(performer_file, dtype=int)  # subject id: 1~9
    label = np.loadtxt(label_file, dtype=int) - 1  # action label: 0~16 (0-based)

    frames_cnt = np.loadtxt(frames_file, dtype=int)  # frames_cnt
    skes_name = np.loadtxt(skes_name_file, dtype=str)
    print('  Done. %d sequences loaded.' % len(skes_name))
    print('  Subjects: %s' % sorted(set(performer)))
    print('  Action classes (0-based): %s' % sorted(set(label)))

    print('[2/6] Loading denoised joints data...')
    with open(raw_skes_joints_pkl, 'rb') as fr:
        skes_joints = pickle.load(fr)  # a list
    print('  Done. %d sequences.' % len(skes_joints))

    print('[3/6] Sequence translation (centering)... This may take a few minutes.')
    skes_joints = seq_translation(skes_joints)
    print('  Done.')

    print('[4/6] Aligning frames to same length...')
    skes_joints = align_frames(skes_joints, frames_cnt)  # aligned to the same frame length
    print('  Done. Shape: %s' % str(skes_joints.shape))

    # HRC only uses Cross-Subject
    evaluation = 'CS'
    print('[5/6] Splitting dataset (%s)...' % evaluation)
    split_dataset(skes_joints, label, performer, evaluation, save_path)
    print('  Saved HRC_%s.npz' % evaluation)

    print('[6/6] Creating aligned dataset (skeleton alignment)... This may take a few minutes.')
    create_aligned_dataset(file_list=['HRC_CS.npz'])
    print('  Saved HRC_CS_aligned.npz')

    print()
    print('All done! Files generated:')
    print('  - HRC_CS.npz')
    print('  - HRC_CS_aligned.npz')
    print()
    print('Next step: create symlink')
    print('  cd data/hrc && ln -sf HRC_CS_aligned.npz CS_aligned.npz')
