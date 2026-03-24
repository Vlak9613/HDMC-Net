"""
HDMC-Net Configuration

Command-line argument parser for HDMC-Net training and evaluation.
"""

import argparse


def str2bool(v):
    """Convert string to boolean."""
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


def get_parser():
    """Get argument parser for HDMC-Net."""
    parser = argparse.ArgumentParser(description='HDMC-Net: Hybrid Dynamic Momentum Causal Network')
    
    # General
    parser.add_argument('--debug', type=str2bool, default=False, help='Debug mode')
    parser.add_argument('--log_dir', type=str, default='.', help='Log directory')
    parser.add_argument('--model_saved_name', default='')
    parser.add_argument('--project', type=str, default='HDMC-Net', help='Project name for wandb')

    # Data
    parser.add_argument('--num_point', type=int, default=25, help='Number of skeleton joints')
    parser.add_argument('--num_person', type=int, default=2, help='Number of persons')
    parser.add_argument('--num_class', type=int, default=60, help='Number of action classes')
    parser.add_argument('--dataset', default='ntu', help='Dataset name')
    parser.add_argument('--datacase', default='CS', help='Data case (CS/CV)')
    parser.add_argument('--use_vel', type=str2bool, default=False, help='Use velocity features')

    # Processor
    parser.add_argument('--phase', default='train', help='train or test')
    parser.add_argument('--save_score', type=str2bool, default=True, help='Save classification scores')

    # Visualization and debug
    parser.add_argument('--seed', type=int, default=1, help='Random seed')
    parser.add_argument('--log_interval', type=int, default=100, help='Log interval')
    parser.add_argument('--save_interval', type=int, default=1, help='Save interval')
    parser.add_argument('--save_epoch', type=int, default=0, help='Start epoch to save model')
    parser.add_argument('--eval_interval', type=int, default=5, help='Evaluation interval')
    parser.add_argument('--print_log', type=str2bool, default=True, help='Print logging')
    parser.add_argument('--show_topk', type=int, default=[1, 5], nargs='+', help='Top K accuracy to show')

    # Feeder
    parser.add_argument('--feeder', default='feeders.ntu_feeder.Feeder', help='Data loader')
    parser.add_argument('--num_worker', type=int, default=8, help='Number of data loader workers')
    parser.add_argument('--random_rot', type=str2bool, default=True, help='Random rotation augmentation')
    parser.add_argument('--train_obs_min', type=float, default=0.5, help='Min observation rate for training')
    parser.add_argument('--train_obs_max', type=float, default=1.0, help='Max observation rate for training')
    parser.add_argument('--test_obs', type=float, default=0.95, help='Observation rate for testing')

    # Model
    parser.add_argument('--window_size', type=int, default=64, help='Temporal window size')
    parser.add_argument('--base_channel', type=int, default=64, help='Base channel dimension')
    parser.add_argument('--weights', default=None, help='Pretrained weights path')
    parser.add_argument('--ignore_weights', type=str, default=[], nargs='+', help='Weights to ignore')
    parser.add_argument('--n_heads', type=int, default=3, help='Number of graph heads')
    parser.add_argument('--depth', type=int, default=4, help='Transformer depth')
    parser.add_argument('--k', type=int, default=8, help='k for adjacency matrix')
    parser.add_argument('--graph', type=str, default='graph.ntu_graph.Graph', help='Graph definition')
    parser.add_argument('--num_cls', type=int, default=1, help='Number of classifiers')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')

    # Optimizer
    parser.add_argument('--base_lr', type=float, default=1e-1, help='Initial learning rate')
    parser.add_argument('--step', type=int, default=[50, 60], nargs='+', help='LR decay epochs')
    parser.add_argument('--optimizer', default='SGD', help='Optimizer type')
    parser.add_argument('--nesterov', type=str2bool, default=True, help='Use Nesterov momentum')
    parser.add_argument('--batch_size', type=int, default=64, help='Training batch size')
    parser.add_argument('--test_batch_size', type=int, default=64, help='Test batch size')
    parser.add_argument('--start_epoch', type=int, default=0, help='Start epoch')
    parser.add_argument('--num_epoch', type=int, default=110, help='Total epochs')
    parser.add_argument('--weight_decay', type=float, default=0.0003, help='Weight decay')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='LR decay rate')
    parser.add_argument('--warm_up_epoch', type=int, default=0, help='Warmup epochs')
    
    # Loss weights
    parser.add_argument('--lambda_1', type=float, default=1e+0, help='Classification loss weight')
    parser.add_argument('--lambda_2', type=float, default=1e-1, help='Reconstruction loss weight')
    parser.add_argument('--lambda_3', type=float, default=1e-2, help='Feature consistency loss weight')
    parser.add_argument('--lambda_cls_guide', type=float, default=0.05, help='Classification guidance loss weight')

    # Training
    parser.add_argument('--half', type=str2bool, default=True, help='Use FP16 training')

    # MomentumNet Extrapolator
    parser.add_argument('--n_step', type=int, default=3, help='Number of prediction steps')
    parser.add_argument('--dilation', type=int, default=1, help='Dilation for temporal convolution')

    return parser
