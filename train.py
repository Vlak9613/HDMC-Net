#!/usr/bin/env python
"""
HDMC-Net Training Script

Train and evaluate HDMC-Net (Hybrid Dynamic Momentum Causal Network) for
online early skeleton-based action recognition.

Usage:
    python train.py --dataset ntu --datacase CS --num_epoch 110
"""
from __future__ import print_function

import os
import sys
import math
import time
import glob
import wandb
import pickle
import random

from collections import OrderedDict

if sys.platform != 'win32':
    import resource
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.cuda.amp import autocast, GradScaler

from tqdm import tqdm

from config import get_parser
from losses import LabelSmoothingCrossEntropy, masked_recon_loss
from model.hdgcn import HDGCN
from utils import AverageMeter, import_class
from einops import rearrange, repeat


def init_seed(seed):
    """Initialize random seeds for reproducibility."""
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


class Processor():
    """Processor for Skeleton-based Action Recognition."""

    def __init__(self, arg):
        self.arg = arg
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.save_arg()
        self.load_model()
        A_vector = self.model.get_A(self.arg.k) if self.arg.k != 8 else None

        self.load_optimizer()
        
        if self.arg.lambda_cls_guide > 0 and hasattr(self, 'cls_guide_head'):
            self.optimizer.add_param_group({
                'params': self.cls_guide_head.parameters(),
                'lr': self.arg.base_lr,
                'weight_decay': self.arg.weight_decay
            })
            print(f"[Processor] cls_guide_head params added to optimizer")
        
        self.load_data(A_vector)

        self.best_acc = 0
        self.best_acc_epoch = 0
        self.best_auc = 0
        self.best_auc_epoch = 0
        self.best_epoch_test_acc = [0.0] * 10
        self.best_model_path = None  # Track best model checkpoint path
        self.log_recon_loss = AverageMeter()
        self.log_auc = AverageMeter()
        self.log_cls_loss = AverageMeter()
        self.log_acc = [AverageMeter() for _ in range(10)]
        self.log_feature_loss = AverageMeter()
        self.log_cls_guide_loss = AverageMeter()
        self.final_train_acc = [0.0] * 10

        self.model = self.model.to(self.device)
        self.scaler = GradScaler(enabled=self.arg.half)

    def load_data(self, A_vector):
        """Load training and testing data."""
        Feeder = import_class(self.arg.feeder)
        self.data_loader = dict()
        data_path = f'data/{self.arg.dataset}/{self.arg.datacase}_aligned.npz'
        self.data_loader['train'] = torch.utils.data.DataLoader(
            dataset=Feeder(data_path=data_path,
                split='train',
                p_interval=[self.arg.train_obs_min, self.arg.train_obs_max],
                vel=self.arg.use_vel,
                random_rot=self.arg.random_rot,
                sort=False,
                A=A_vector,
                window_size=self.arg.window_size,
            ),
            batch_size=self.arg.batch_size,
            shuffle=True,
            num_workers=self.arg.num_worker,
            drop_last=True,
            pin_memory=True,
            worker_init_fn=init_seed)

        self.data_loader['test'] = torch.utils.data.DataLoader(
            dataset=Feeder(
                data_path=data_path,
                split='test',
                p_interval=[self.arg.test_obs],
                vel=self.arg.use_vel,
                A=A_vector,
                window_size=self.arg.window_size,
            ),
            batch_size=self.arg.test_batch_size,
            shuffle=False,
            num_workers=self.arg.num_worker,
            drop_last=False,
            pin_memory=True,
            worker_init_fn=init_seed)

    def load_model(self):
        """Load HDMC-Net model."""
        self.model = HDGCN(
            num_class=self.arg.num_class,
            num_point=self.arg.num_point,
            num_person=self.arg.num_person,
            graph=self.arg.graph,
            in_channels=3,
            num_head=self.arg.n_heads,
            k=self.arg.k,
            base_channel=self.arg.base_channel,
            depth=self.arg.depth,
            device=self.device,
            T=self.arg.window_size,
            n_step=self.arg.n_step,
            dilation=self.arg.dilation,
            num_cls=self.arg.num_cls,
            dropout=self.arg.dropout,
        )
        self.cls_loss = LabelSmoothingCrossEntropy(T=self.arg.window_size).to(self.device)
        self.recon_loss = masked_recon_loss
        
        if self.arg.lambda_cls_guide > 0 and self.arg.n_step > 0:
            z_hat_dim = self.arg.base_channel * self.arg.n_step
            self.cls_guide_head = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(z_hat_dim, z_hat_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(z_hat_dim // 2, self.arg.num_class)
            ).to(self.device)
            print(f"[Processor] Classification Guidance enabled with lambda={self.arg.lambda_cls_guide}")
        elif self.arg.n_step == 0:
            print(f"[Processor] n_step=0 ablation mode: recon/feature/cls_guide losses disabled")

        if self.arg.weights:
            self.print_log('Load weights from {}.'.format(self.arg.weights))
            if '.pkl' in self.arg.weights:
                with open(self.arg.weights, 'r') as f:
                    weights = pickle.load(f)
            else:
                weights = torch.load(self.arg.weights)

            weights = OrderedDict([[k.split('module.')[-1], v.to(self.device)] for k, v in weights.items()])

            keys = list(weights.keys())
            for w in self.arg.ignore_weights:
                for key in keys:
                    if w in key:
                        if weights.pop(key, None) is not None:
                            self.print_log('Successfully Remove Weights: {}.'.format(key))
                        else:
                            self.print_log('Can Not Remove Weights: {}.'.format(key))

            try:
                self.model.load_state_dict(weights, strict=False)
            except:
                state = self.model.state_dict()
                diff = list(set(state.keys()).difference(set(weights.keys())))
                print('Can not find these weights:')
                for d in diff:
                    print('  ' + d)
                state.update(weights)
                self.model.load_state_dict(state)

    def load_optimizer(self):
        """Load optimizer."""
        if self.arg.optimizer == 'SGD':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.arg.base_lr,
                momentum=0.9,
                nesterov=self.arg.nesterov,
                weight_decay=self.arg.weight_decay)
        elif self.arg.optimizer == 'Adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.arg.base_lr,
                weight_decay=self.arg.weight_decay)
        else:
            raise ValueError()

        self.print_log('using warm up, epoch: {}'.format(self.arg.warm_up_epoch))

    def save_arg(self):
        """Save arguments."""
        arg_dict = vars(self.arg)
        if not os.path.exists(self.arg.work_dir):
            os.makedirs(self.arg.work_dir)

    def save_best_model(self, epoch):
        """Save model checkpoint only when best AUC is achieved.
        Deletes previous best checkpoint to save disk space."""
        save_path = os.path.join(self.arg.work_dir, f'best_model_epoch{epoch+1}.pt')
        state = {
            'epoch': epoch + 1,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_auc': self.best_auc,
            'best_epoch_test_acc': self.best_epoch_test_acc,
        }
        if hasattr(self, 'cls_guide_head'):
            state['cls_guide_head_state_dict'] = self.cls_guide_head.state_dict()
        torch.save(state, save_path)
        self.print_log(f'    Model saved to {save_path}')
        # Delete previous best checkpoint
        if self.best_model_path and os.path.exists(self.best_model_path) and self.best_model_path != save_path:
            os.remove(self.best_model_path)
            self.print_log(f'    Removed old checkpoint: {self.best_model_path}')
        self.best_model_path = save_path

    def cleanup_wandb_checkpoints(self):
        """Remove wandb internal runs-*.pt checkpoints to save disk space."""
        if not hasattr(self, 'arg') or not hasattr(self.arg, 'work_dir'):
            return
        wandb_run_dir = self.arg.work_dir  # e.g. wandb/offline-run-.../files
        pt_files = glob.glob(os.path.join(wandb_run_dir, 'runs-*.pt'))
        if pt_files:
            removed_size = 0
            for f in pt_files:
                removed_size += os.path.getsize(f)
                os.remove(f)
            self.print_log(f'    Cleaned up {len(pt_files)} wandb checkpoint files '
                           f'({removed_size / 1024 / 1024:.1f} MB freed)')

    def adjust_learning_rate(self, epoch):
        """Adjust learning rate with warmup and step decay."""
        if self.arg.optimizer == 'SGD' or self.arg.optimizer == 'Adam':
            if epoch < self.arg.warm_up_epoch and self.arg.weights is None:
                lr = self.arg.base_lr * (epoch + 1) / self.arg.warm_up_epoch
            else:
                lr = self.arg.base_lr * (
                        self.arg.lr_decay_rate ** np.sum(epoch >= np.array(self.arg.step)))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            return lr
        else:
            raise ValueError()

    def print_time(self):
        """Print current time."""
        localtime = time.asctime(time.localtime(time.time()))
        self.print_log("Local current time :  " + localtime)

    def print_log(self, str, print_time=True):
        """Print log message."""
        if print_time:
            localtime = time.asctime(time.localtime(time.time()))
            str = "[ " + localtime + ' ] ' + str
        print(str)
        if self.arg.print_log:
            with open('{}/log.txt'.format(self.arg.work_dir), 'a') as f:
                print(str, file=f)

    def train(self, epoch, save_model=False):
        """Train for one epoch."""
        self.model.train()
        [self.log_acc[i].reset() for i in range(10)]
        self.log_cls_loss.reset()
        self.log_auc.reset()
        self.log_feature_loss.reset()
        self.log_recon_loss.reset()
        self.log_cls_guide_loss.reset()
        self.print_log('Training epoch: {}'.format(epoch + 1))
        lr = self.adjust_learning_rate(epoch)
        tbar = tqdm(self.data_loader['train'], dynamic_ncols=True)

        for x, y, mask, index in tbar:
            cls_loss, recon_loss, feature_loss, cls_guide_loss = torch.tensor(0.), torch.tensor(0.), torch.tensor(0.), torch.tensor(0.)
            B, C, T, V, M = x.shape
            x = x.float().to(self.device)
            y = y.long().to(self.device)
            y_original = y.clone()
            mask = mask.long().to(self.device)
            
            with autocast(enabled=self.arg.half):
                y_hat, x_hat, z_0, z_hat, _ = self.model(x)
                N_cls = y_hat.size(0) // B

                if self.arg.lambda_1:
                    y = y.view(1, B, 1).expand(N_cls, B, y_hat.size(2))
                    y_hat_ = rearrange(y_hat, "b i t -> (b t) i")
                    cls_loss = self.arg.lambda_1 * self.cls_loss(y_hat_, y.reshape(-1))

                if self.arg.lambda_2 and self.arg.n_step > 0:
                    N_rec = x_hat.size(0) // B
                    x_gt = x.unsqueeze(0).expand(N_rec, B, C, T, V, M).reshape(N_rec*B, C, T, V, M)
                    mask_recon = repeat(mask, 'b c t v m -> n b c t v m', n=N_rec).clone()
                    for i in range(N_rec):
                        if N_rec == self.arg.n_step:
                            mask_recon[i, :, :, :i+1, :, :] = 0.
                        else:
                            mask_recon[i, :, :, :i, :, :] = 0.
                    mask_recon = rearrange(mask_recon, 'n b c t v m -> (n b) c t v m')
                    recon_loss = self.arg.lambda_2 * self.recon_loss(x_hat, x_gt, mask_recon)

                if self.arg.lambda_3 and self.arg.n_step > 0:
                    N_step = self.arg.n_step
                    B_, C_, T_, V_ = z_0.shape
                    z_hat_reshaped = z_hat.view(N_step, B_, C_, T_, V_)
                    z_0_last = z_0[:, :, -1:, :]
                    z_hat_last_frames = z_hat_reshaped[:, :, :, -1:, :]
                    z_0_last_expanded = repeat(z_0_last, 'b c t v -> n b c t v', n=N_step)
                    feature_loss = self.arg.lambda_3 * F.mse_loss(z_hat_last_frames, z_0_last_expanded)

                if self.arg.lambda_cls_guide > 0 and self.arg.n_step > 0:
                    N_step = self.arg.n_step
                    B_, C_, T_, V_ = z_0.shape
                    z_hat_cls = rearrange(z_hat, '(n b) c t v -> b (n c) t v', n=N_step)
                    z_hat_first = z_hat_cls[:, :C_, :, :]
                    z_guide_input = torch.cat([z_hat_first, z_hat_cls], dim=1)
                    z_guide_decoded = self.model.cls_decoder(z_guide_input)
                    z_guide_pooled = self.model.spatial_pooling(z_guide_decoded, dim=-1)
                    y_guide_logits = self.model.classifier_lst[-1](z_guide_pooled[:, :, -1:])
                    y_guide_logits = y_guide_logits.squeeze(-1)
                    y_guide = y_original.unsqueeze(1).expand(-1, M).reshape(-1)
                    cls_guide_loss = self.arg.lambda_cls_guide * F.cross_entropy(y_guide_logits, y_guide)

                loss = cls_loss + recon_loss + feature_loss + cls_guide_loss

            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            value, predict_label = torch.max(y_hat.data, 1)
            for i, ratio in enumerate([(i+1)/10 for i in range(10)]):
                self.log_acc[i].update((predict_label == y.data)
                                        .view(N_cls*B, -1)[:, int(math.ceil(T*ratio))-1].float().mean(), B)
            self.log_cls_loss.update(cls_loss.data.item(), B)
            self.log_feature_loss.update(feature_loss.data.item(), B)
            self.log_recon_loss.update(recon_loss.data.item(), B)
            self.log_cls_guide_loss.update(cls_guide_loss.data.item() if torch.is_tensor(cls_guide_loss) else cls_guide_loss, B)

            AUC = np.mean([self.log_acc[i].avg.cpu().numpy() for i in range(10)])
            ft_raw = self.log_feature_loss.avg / self.arg.lambda_3 if self.arg.lambda_3 > 0 else 0
            tbar.set_description(
                f"[Epoch #{epoch}] "
                f"AUC:{AUC:.3f}, "
                f"CLS:{self.log_cls_loss.avg:.3f}, "
                f"FT:{ft_raw:.4f}, "
                f"CG:{self.log_cls_guide_loss.avg:.3f}, "
                f"RECON:{self.log_recon_loss.avg:.5f}"
            )
            
        AUC = np.mean([self.log_acc[i].avg.cpu().numpy() for i in range(10)])
        train_dict = {
            "train/Recon_loss": self.log_recon_loss.avg,
            "train/cls_loss": self.log_cls_loss.avg,
            "train/feature_loss": self.log_feature_loss.avg,
            "train/cls_guide_loss": self.log_cls_guide_loss.avg,
            "train/AUC": AUC,
        }
        train_dict.update({f"train/ACC_{(i+1)/10}": self.log_acc[i].avg for i in range(10)})
        wandb.log(train_dict)
        
        self.print_log(f'  [Train Epoch {epoch+1}] Accuracy at each observation rate:')
        acc_str = '    '
        for i in range(10):
            acc_val = self.log_acc[i].avg
            if hasattr(acc_val, 'cpu'):
                acc_val = acc_val.cpu().numpy()
            acc_str += f'{(i+1)*10}%:{acc_val:.4f}  '
        self.print_log(acc_str)
        self.print_log(f'    AUC: {AUC:.4f}')
        
        for i in range(10):
            acc_val = self.log_acc[i].avg
            self.final_train_acc[i] = acc_val.cpu().numpy() if hasattr(acc_val, 'cpu') else acc_val
            
        # Checkpoint saving moved to eval() — only best AUC model is saved

    def eval(self, epoch, save_score=False, loader_name=['test']):
        """Evaluate model."""
        self.model.eval()
        [self.log_acc[i].reset() for i in range(10)]
        self.log_cls_loss.reset()
        self.log_feature_loss.reset()
        self.log_recon_loss.reset()
        self.print_log('Eval epoch: {}'.format(epoch + 1))
        
        for ln in loader_name:
            loss_value = []
            score_frag = []
            label_list = []
            tbar = tqdm(self.data_loader[ln], dynamic_ncols=True)
            
            for x, y, mask, index in tbar:
                label_list.append(y)
                with torch.no_grad():
                    B, C, T, V, M = x.shape
                    x = x.float().to(self.device)
                    y = y.long().to(self.device)
                    mask = mask.long().to(self.device)
                    
                    y_hat, x_hat, z_0, z_hat, _ = self.model(x)
                    N_cls = y_hat.size(0) // B
                    y = y.view(1, B, 1).expand(N_cls, B, y_hat.size(2)).reshape(-1)
                    y_hat = rearrange(y_hat, "b i t -> (b t) i")
                    cls_loss = self.cls_loss(y_hat, y)

                    recon_loss = torch.tensor(0.)
                    feature_loss = torch.tensor(0.)

                    if self.arg.n_step > 0:
                        N_rec = x_hat.size(0) // B
                        x_gt = x.unsqueeze(0).expand(N_rec, B, C, T, V, M).reshape(N_rec*B, C, T, V, M)
                        mask_recon = repeat(mask, 'b c t v m -> n b c t v m', n=N_rec)
                        for i in range(N_rec):
                            if N_rec == self.arg.n_step:
                                mask_recon[i, :, :, :i+1, :, :] = 0.
                            else:
                                mask_recon[i, :, :, :i, :, :] = 0.
                        mask_recon = rearrange(mask_recon, 'n b c t v m -> (n b) c t v m')
                        recon_loss = self.recon_loss(x_hat, x_gt, mask_recon)

                        N_step = self.arg.n_step
                        B_, C_, T_, V_ = z_0.shape
                        z_0_exp = repeat(z_0, 'b c t v-> n b c t v', n=N_step)
                        z_hat_reshaped = z_hat.view(N_step, B_, C_, T_, V_)
                        mask_feature = (z_hat_reshaped != 0.)
                        feature_loss = self.recon_loss(z_0_exp, z_hat_reshaped, mask_feature)

                    loss = self.arg.lambda_2 * recon_loss + self.arg.lambda_1 * cls_loss
                    score_frag.append(y_hat.view(B, T, -1)[:, :, :].data.cpu().numpy())
                    loss_value.append(loss.data.item())

                    _, predict_label = torch.max(y_hat.data, 1)
                    
                for i, ratio in enumerate([(i+1)/10 for i in range(10)]):
                    self.log_acc[i].update((predict_label == y.data)
                                            .view(N_cls*B, -1)[:, int(math.ceil(T*ratio))-1].float().mean(), B)
                self.log_auc.update((predict_label == y.data)
                                    .view(N_cls, B, -1)[-1, :, :].float().mean(), B)
                self.log_cls_loss.update(cls_loss.data.item(), B)
                self.log_recon_loss.update(recon_loss.data.item(), B)
                self.log_feature_loss.update(feature_loss.data.item(), B)

                AUC = np.mean([self.log_acc[i].avg.cpu().numpy() for i in range(10)])
                tbar.set_description(
                    f"[Epoch #{epoch}] "
                    f"AUC:{AUC:.3f}, "
                    f"CLS:{self.log_cls_loss.avg:.3f}, "
                    f"FT:{self.log_feature_loss.avg:.3f}, "
                    f"RECON:{self.log_recon_loss.avg:.5f}"
                )
                
            AUC = np.mean([self.log_acc[i].avg.cpu().numpy() for i in range(10)])
            eval_dict = {
                "eval/Recon_loss": self.log_recon_loss.avg,
                "eval/cls_loss": self.log_cls_loss.avg,
                "eval/feature_loss": self.log_feature_loss.avg,
                "eval/AUC": AUC,
            }
            eval_dict.update({f"eval/ACC_{(i+1)/10}": self.log_acc[i].avg for i in range(10)})
            wandb.log(eval_dict)
            
            self.print_log(f'  [Eval Epoch {epoch+1}] Accuracy at each observation rate:')
            acc_str = '    '
            current_acc = []
            for i in range(10):
                acc_val = self.log_acc[i].avg
                if hasattr(acc_val, 'cpu'):
                    acc_val = acc_val.cpu().numpy()
                acc_str += f'{(i+1)*10}%:{acc_val:.4f}  '
                current_acc.append(acc_val)
            self.print_log(acc_str)
            self.print_log(f'    AUC: {AUC:.4f}')
            
            if AUC > self.best_auc:
                self.best_auc = AUC
                self.best_auc_epoch = epoch + 1
                self.best_epoch_test_acc = current_acc.copy()
                self.print_log(f'    *** New Best AUC: {AUC:.4f} at Epoch {epoch+1} ***')
                self.save_best_model(epoch)

            score = np.concatenate(score_frag)

            if 'ucla' in self.arg.feeder:
                self.data_loader[ln].dataset.sample_name = np.arange(len(score))

            score_dict = dict(zip(self.data_loader[ln].dataset.sample_name, score))

            # Only save score for the best AUC epoch (saved in save_best_model)
            if save_score and AUC >= self.best_auc:
                score_path = os.path.join(self.arg.work_dir, f'best_epoch{epoch+1}_{ln}_score.pkl')
                # Remove previous best score file
                old_scores = glob.glob(os.path.join(self.arg.work_dir, f'best_epoch*_{ln}_score.pkl'))
                for old in old_scores:
                    if old != score_path:
                        os.remove(old)
                with open(score_path, 'wb') as f:
                    pickle.dump(score_dict, f)

    def start(self):
        """Start training or testing."""
        if self.arg.phase == 'train':
            self.print_log('Parameters:\n{}\n'.format(str(vars(self.arg))))
            def count_parameters(model):
                return sum(p.numel() for p in model.parameters() if p.requires_grad)
            self.print_log(f'# Parameters: {count_parameters(self.model)/10**6:.3f}M')
            
            for epoch in range(self.arg.start_epoch, self.arg.num_epoch):
                save_model = (epoch + 1 > self.arg.save_epoch)
                self.train(epoch, save_model=save_model)
                if epoch > self.arg.save_epoch:
                    self.eval(epoch, save_score=self.arg.save_score, loader_name=['test'])
            self.arg.print_log = True

            num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            
            self.print_log('='*70)
            self.print_log(f'BEST Results (Epoch {self.best_auc_epoch} with Best AUC {self.best_auc:.4f}):')
            self.print_log('-'*70)
            self.print_log(f'  {"Obs Rate":<12} {"Test ACC":<15}')
            self.print_log('-'*70)
            for i in range(10):
                self.print_log(f'  {(i+1)*10:3d}%         {self.best_epoch_test_acc[i]:<15.4f}')
            self.print_log('-'*70)
            self.print_log(f'  {"AUC":<12} {self.best_auc:<15.4f}')
            self.print_log('='*70)
            
            self.print_log(f'Best AUC: {self.best_auc:.4f} (Epoch {self.best_auc_epoch})')
            if self.best_model_path:
                self.print_log(f'Best model saved at: {self.best_model_path}')
            self.print_log(f'Model name: {self.arg.work_dir}')
            self.print_log(f'Model total number of params: {num_params}')
            self.print_log(f'Weight decay: {self.arg.weight_decay}')
            self.print_log(f'Base LR: {self.arg.base_lr}')
            self.print_log(f'Batch Size: {self.arg.batch_size}')
            self.print_log(f'seed: {self.arg.seed}')
            
            # Clean up wandb internal checkpoint files to save disk space
            self.cleanup_wandb_checkpoints()

        elif self.arg.phase == 'test':
            if self.arg.weights is None:
                raise ValueError('Please appoint --weights.')
            self.arg.print_log = False
            self.eval(epoch=0, save_score=self.arg.save_score, loader_name=['test'])
            self.print_log('Done.\n')


def main():
    """Main entry point."""
    parser = get_parser()
    arg = parser.parse_args()
    wandb.init(
        project=arg.project,
        mode="offline",
        settings=wandb.Settings(
            _disable_stats=True,       # Disable system stats collection
            console="off",             # Don't capture console output
        ),
    )
    # Disable wandb auto model checkpointing
    wandb.save_policy = "end"
    arg.work_dir = wandb.run.dir
    wandb.config.update(arg)
    init_seed(arg.seed)
    processor = Processor(arg)
    processor.start()


if __name__ == '__main__':
    main()
