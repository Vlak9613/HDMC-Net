"""
HRC-Assembly Dataset — Data Conversion Script

Converts raw .npy skeleton files (single person, shape=(T, 75) each)
collected by collect_skeleton.py into intermediate formats for HDMC-Net training.

Usage:
  cd data/hrc
  python build_dataset.py --raw_dir ./raw_skeletons

Input:
  ./raw_skeletons/P01_A01_R01.npy    # (T, 75) per sequence, single person 25 joints x 3 coords
  ./raw_skeletons/P01_A01_R02.npy
  ...

Output:
  ./denoised_data/raw_denoised_joints.pkl   # list of (T_i, 75) arrays
  ./denoised_data/frames_cnt.txt            # frame count per sequence
  ./statistics/skes_available_name.txt      # sequence name list
  ./statistics/label.txt                    # action labels (1-based)
  ./statistics/performer.txt                # subject IDs
  ./statistics/replication.txt              # repetition indices
  ./statistics/camera.txt                   # camera IDs (all 1)
"""

import os
import glob
import argparse
import numpy as np
import pickle


def parse_filename(filename):
    """
    Parse filename: P{performer:02d}_A{action:02d}_R{replication:02d}.npy

    Returns:
        performer (int), action (int), replication (int)  all 1-based
    """
    basename = os.path.splitext(os.path.basename(filename))[0]
    parts = basename.split('_')
    performer = int(parts[0][1:])
    action = int(parts[1][1:])
    replication = int(parts[2][1:])
    return performer, action, replication


def build_statistics(npy_files, stat_dir):
    """Generate statistics files."""
    os.makedirs(stat_dir, exist_ok=True)

    skes_names = []
    labels = []
    performers = []
    replications = []
    cameras = []

    for f in npy_files:
        basename = os.path.splitext(os.path.basename(f))[0]
        performer, action, replication = parse_filename(f)

        skes_names.append(basename)
        labels.append(action)           # 1-based
        performers.append(performer)
        replications.append(replication)
        cameras.append(1)               # single camera

    np.savetxt(os.path.join(stat_dir, 'skes_available_name.txt'), skes_names, fmt='%s')
    np.savetxt(os.path.join(stat_dir, 'label.txt'), labels, fmt='%d')
    np.savetxt(os.path.join(stat_dir, 'performer.txt'), performers, fmt='%d')
    np.savetxt(os.path.join(stat_dir, 'replication.txt'), replications, fmt='%d')
    np.savetxt(os.path.join(stat_dir, 'camera.txt'), cameras, fmt='%d')

    print(f'  Statistics saved to: {stat_dir}')
    print(f'  Sequences: {len(skes_names)}')
    print(f'  Subjects: {sorted(set(performers))}')
    print(f'  Action classes: {sorted(set(labels))}')
    print(f'  Repetitions: {sorted(set(replications))}')


def build_denoised_data(npy_files, denoised_dir):
    """
    Merge .npy files into raw_denoised_joints.pkl

    Each .npy has shape = (T, 75), single person, stored directly into pkl list.
    The align_frames function in seq_transformation.py will zero-pad to 150 dims.
    """
    os.makedirs(denoised_dir, exist_ok=True)

    raw_denoised_joints = []
    frames_cnt = []

    for f in npy_files:
        data = np.load(f).astype(np.float32)  # (T, 75)

        if data.ndim != 2:
            print(f'  Warning: Skipping file with abnormal dimensions {os.path.basename(f)}, shape={data.shape}')
            continue

        if data.shape[1] == 150:
            # Compatible: accept 150-dim data (legacy format)
            pass
        elif data.shape[1] == 75:
            # Standard single-person data
            pass
        else:
            print(f'  Warning: Skipping file {os.path.basename(f)}, expected 75 dims per frame, got {data.shape[1]}')
            continue

        raw_denoised_joints.append(data)
        frames_cnt.append(data.shape[0])

    pkl_path = os.path.join(denoised_dir, 'raw_denoised_joints.pkl')
    with open(pkl_path, 'wb') as fw:
        pickle.dump(raw_denoised_joints, fw, pickle.HIGHEST_PROTOCOL)

    cnt_path = os.path.join(denoised_dir, 'frames_cnt.txt')
    np.savetxt(cnt_path, frames_cnt, fmt='%d')

    frames_cnt = np.array(frames_cnt)
    print(f'  Skeleton data saved to: {pkl_path}')
    print(f'  Sequences: {len(raw_denoised_joints)}')
    print(f'  Frame count range: {frames_cnt.min()} ~ {frames_cnt.max()}')
    print(f'  Average frames: {frames_cnt.mean():.1f}')
    print(f'  Total frames: {frames_cnt.sum()}')


def validate_data(npy_files):
    """Validate data integrity."""
    print('\n[Validation] Checking data integrity...')

    all_performers = set()
    all_actions = set()
    all_reps = set()
    issues = []

    for f in npy_files:
        basename = os.path.basename(f)
        try:
            performer, action, replication = parse_filename(f)
            all_performers.add(performer)
            all_actions.add(action)
            all_reps.add(replication)

            data = np.load(f)
            if data.shape[1] not in (75, 150):
                issues.append(f'  Dimension error: {basename}, shape={data.shape}, expected 75 dims per frame')
            if data.shape[0] < 10:
                issues.append(f'  Too few frames: {basename}, only {data.shape[0]} frames')
            if np.any(np.isnan(data)):
                issues.append(f'  Contains NaN: {basename}')
            if np.all(data[:, :75] == 0):
                issues.append(f'  All-zero data: {basename}')
        except Exception as e:
            issues.append(f'  File error: {basename}, {e}')

    # Check for missing combinations
    expected = len(all_performers) * len(all_actions) * len(all_reps)
    actual = len(npy_files)
    if actual < expected:
        existing = {os.path.splitext(os.path.basename(f))[0] for f in npy_files}
        missing = []
        for p in all_performers:
            for a in all_actions:
                for r in all_reps:
                    name = f'P{p:02d}_A{a:02d}_R{r:02d}'
                    if name not in existing:
                        missing.append(name)
        if missing:
            issues.append(f'  Missing files: expected {expected}, actual {actual}')
            for m in missing[:10]:
                issues.append(f'    Missing: {m}.npy')
            if len(missing) > 10:
                issues.append(f'    ... {len(missing)} missing in total')

    if issues:
        print('  Issues found:')
        for issue in issues:
            print(issue)
    else:
        print('  All checks passed.')

    return len(issues) == 0


def main(args):
    raw_dir = args.raw_dir
    stat_dir = args.stat_dir
    denoised_dir = args.denoised_dir

    print('=' * 60)
    print('  HRC-Assembly Dataset Builder')
    print('=' * 60)

    npy_files = sorted(glob.glob(os.path.join(raw_dir, 'P*_A*_R*.npy')))

    if len(npy_files) == 0:
        print(f'\nError: No .npy files found in {raw_dir}!')
        print('Please run collect_skeleton.py to collect data first.')
        return

    print(f'\nFound {len(npy_files)} skeleton sequence files')

    validate_data(npy_files)

    print('\n[1/2] Generating statistics files...')
    build_statistics(npy_files, stat_dir)

    print('\n[2/2] Building skeleton data file...')
    build_denoised_data(npy_files, denoised_dir)

    print('\n' + '=' * 60)
    print('  Build complete!')
    print('  Next step: python seq_transformation.py')
    print('=' * 60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='HRC-Assembly Data Conversion')
    parser.add_argument('--raw_dir', type=str, default='./raw_skeletons',
                        help='Directory containing raw .npy files')
    parser.add_argument('--stat_dir', type=str, default='./statistics',
                        help='Directory to save statistics files')
    parser.add_argument('--denoised_dir', type=str, default='./denoised_data',
                        help='Directory to save processed data')
    args = parser.parse_args()
    main(args)
