import argparse
import numpy as np
from tqdm import tqdm
import math
import sys
import os
import matplotlib.pyplot as plt
import torch.optim as optim

# torch
import torch
import torch.nn.functional as F

# torchlight
from torchlight import str2bool

from .processor import Processor
from .pretrain import PT_Processor
from .knn_monitor import knn_predict
from net.generic import concat_all_gather


def clip_gradients(model, clip):
    for name, p in model.named_parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            clip_coef = clip / (param_norm + 1e-6)
            if clip_coef < 1:
                p.grad.data.mul_(clip_coef)


class CLRTuningProcessor(PT_Processor):
    """
        Processor for CLR Tuning.
    """
    
    def load_optimizer(self):
        parameters = []
        for name, param in self.model.encoder_student.named_parameters():
            appended = False
            if self.arg.third_stage:
                if name.startswith('blocks'):
                    block_idx = int(name.split('.')[1])
                    if block_idx >= self.arg.model_args['depth'] // 2 or self.arg.fully_tune:
                        decayed_lr = self.arg.base_lr *\
                                    (self.arg.layer_decay ** (self.arg.model_args['depth'] - block_idx))
                        parameters.append({'params': param, 'lr': decayed_lr})
                        appended = True
                if name.startswith('norm') or name.startswith('head'):
                    parameters.append({'params': param, 'lr': self.arg.base_lr})
                    appended = True
            elif name.startswith('head'):
                parameters.append({'params': param, 'lr': self.arg.base_lr})
                appended = True
                
            if not appended:
                param.required_grad = False
        
        self.io.print_log(f"# learnable parameters: {len(parameters)}")

        if self.arg.optimizer == 'SGD':
            self.optimizer = optim.SGD(
                parameters,
                lr=self.arg.base_lr,
                momentum=0.9,
                nesterov=self.arg.nesterov,
                weight_decay=self.arg.weight_decay)
        elif self.arg.optimizer == 'Adam':
            self.optimizer = optim.Adam(
                parameters,
                lr=self.arg.base_lr,
                weight_decay=self.arg.weight_decay)
        elif self.arg.optimizer == "AdamW":
            self.optimizer = optim.AdamW(
                parameters,
                lr=self.arg.base_lr,
                betas=(0.9, 0.95)
            )
        else:
            raise ValueError()
        
    def step_decay(self, epoch):
        if epoch in self.arg.step_epochs:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] *= self.arg.lr_decay

    def train(self, epoch):
        self.model.train()
        loader = self.data_loader['train']
        loss_value = []

        self.step_decay(epoch)

        for it, (all_data, indices) in enumerate(loader):
            it = len(loader) * epoch + it
            self.global_step += 1
            # get data
            for i in range(len(all_data)):
                data = all_data[i]
                data = data.float().to(self.dev, non_blocking=True)
                data = self.view_gen(data)
                all_data[i] = data

            # forward
            loss = self.model(all_data[0], all_data[1], indices)

            if not math.isfinite(loss.item()):
                print(f"Loss is {loss.item()}, stopping training", flush=True)
                sys.exit(1)

            # backward
            self.optimizer.zero_grad()
            loss.backward()
            clip_gradients(self.model.module.encoder_student, self.arg.clip_grad)
            self.optimizer.step()

            # statistics
            self.iter_info['loss'] = loss.data.item()
            loss_value.append(self.iter_info['loss'])
            if self.local_rank == 0:
                self.show_iter_info()
            self.meta_info['iter'] += 1
                    
        self.epoch_info['train_mean_loss']= np.mean(loss_value)
        if self.local_rank == 0:
            self.show_epoch_info()
        
    def view_gen(self, data):
        if self.arg.stream == 'joint':
            pass
        elif self.arg.stream == 'motion':
            motion = torch.zeros_like(data)

            motion[:, :, :-1, :, :] = data[:, :, 1:, :, :] - data[:, :, :-1, :, :]

            data = motion
        elif self.arg.stream == 'bone':
            Bone = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6), (8, 7), (9, 21),
                    (10, 9), (11, 10), (12, 11), (13, 1), (14, 13), (15, 14), (16, 15), (17, 1),
                    (18, 17), (19, 18), (20, 19), (21, 21), (22, 23), (23, 8), (24, 25), (25, 12)]

            bone = torch.zeros_like(data)

            for v1, v2 in Bone:
                bone[:, :, :, v1 - 1, :] = data[:, :, :, v1 - 1, :] - data[:, :, :, v2 - 1, :]
                bone[:, :, :, v1 - 1, :] = data[:, :, :, v1 - 1, :] - data[:, :, :, v2 - 1, :]

            data = bone
        else:
            raise ValueError

        return data
    
    @torch.no_grad()
    def knn_monitor(self, epoch):
        self.model.module.encoder_student.eval()
        feature_bank, label_bank = [], []
        with torch.no_grad():
            # generate feature bank
            for data, label in tqdm(self.data_loader['mem_train'], desc='Feature extracting'):
                data = data.float().to(self.dev, non_blocking=True)
                label = label.long().to(self.dev, non_blocking=True)

                data = self.view_gen(data)
                if self.arg.third_stage:
                    feature = self.model.module.encoder_student(data, return_feature=True)
                else: # Second stage signal is performance of prediction head
                    feature, _ = self.model.module.encoder_student(data)
                feature = F.normalize(feature, dim=1)
                feature_bank.append(feature)
                label_bank.append(label)
            # [D, N]
            feature_bank = concat_all_gather(torch.cat(feature_bank, dim=0)).t().contiguous()
            # [N]
            feature_labels = concat_all_gather(torch.cat(label_bank)).to(feature_bank.device)
            # loop test data to predict the label by weighted knn search
            for i in self.arg.knn_k:
                total_top1, total_top5, total_num = 0, 0, 0
                test_bar = tqdm(self.data_loader['mem_test'], desc='kNN-{}'.format(i))
                for data, label in test_bar:
                    data = data.float().to(self.dev, non_blocking=True)
                    label = label.float().to(self.dev, non_blocking=True)

                    data = self.view_gen(data)

                    if self.arg.third_stage:
                        feature = self.model.module.encoder_student(data, return_feature=True)
                    else:
                        feature, _ = self.model.module.encoder_student(data)
                    feature = concat_all_gather(F.normalize(feature, dim=1))

                    pred_labels = knn_predict(feature, feature_bank, feature_labels, self.arg.knn_classes, i,
                                              self.arg.knn_t)

                    total_num += feature.size(0)
                    total_top1 += (pred_labels[:, 0] == concat_all_gather(label)).float().sum().item()
                    test_bar.set_postfix({'k': i, 'Accuracy': total_top1 / total_num * 100})
                acc = total_top1 / total_num * 100

                self.knn_results[i][epoch] = acc
    
    @staticmethod
    def get_parser(add_help=False):

        # parameter priority: command line > config > default
        parent_parser = Processor.get_parser(add_help=False)
        parser = argparse.ArgumentParser(
            add_help=add_help,
            parents=[parent_parser],
            description='Spatial Temporal Graph Convolution Network')

        parser.add_argument('--base_lr', type=float, default=0.01, help='initial learning rate')
        parser.add_argument('--lr_decay', type=float, default=0.75, help='Learning rate decay')
        parser.add_argument('--fully_tune', type=str2bool, default=False, help='Finetune fully or half of it')
        parser.add_argument('--step_epochs', type=int, default=[], nargs='+', help='the epoch where optimizer reduce the learning rate')
        parser.add_argument('--optimizer', default='SGD', help='type of optimizer')
        parser.add_argument('--nesterov', type=str2bool, default=True, help='use nesterov or not')
        parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay for optimizer')
        parser.add_argument('--stream', type=str, default='joint', help='the stream of input')
        parser.add_argument('--loss', default='cross-entropy', help='Loss function')
        parser.add_argument('--momentum_teacher', type=float, default=0.996, help='Momentum used for updating the weights of teacher')
        parser.add_argument('--warmup_teacher_temp', type=float, default=0.01, help='Teacher temperature in warmup stage')
        parser.add_argument('--warmup_teacher_temp_epochs', type=int, default=0, help='Number of epochs for warmup of teacher temperature')
        parser.add_argument('--teacher_temp', type=float, default=0.01, help='Teacher temperature')
        parser.add_argument('--student_temp', type=float, default=0.02, help='Student temperature')
        parser.add_argument('--centering_type', default='sinkhorn_knopp', help='Type of centering on student network')
        parser.add_argument('--clip_grad', type=float, default=3., help='Clip gradient value')
        parser.add_argument('--draw_distribution', type=str2bool, default=True, help='Draw output distribution for some samples')
        parser.add_argument('--draw_freq', type=int, default=10, help="Draw distribution's frequency")
        parser.add_argument('--selected_indices', type=int, default=[], nargs='+', help='Indices from dataset to draw')
        parser.add_argument('--koleo_weight', type=int, default=0.1, help="Koleo Loss weight")
        parser.add_argument('--third_stage', type=str2bool, default=False, help="Third stage tunes half of the encoder in addition to head")
        parser.add_argument('--layer_decay', type=int, default=0.65, help="layer-wise lr decay (For 3rd stage)")
        return parser
    
    @staticmethod
    def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0):
        warmup_schedule = np.array([])
        warmup_iters = warmup_epochs * niter_per_ep
        if warmup_epochs > 0:
            warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

        iters = np.arange(epochs * niter_per_ep - warmup_iters)
        schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))

        schedule = np.concatenate((warmup_schedule, schedule))
        assert len(schedule) == epochs * niter_per_ep
        return schedule
