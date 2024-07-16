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


class DINOTuningProcessor(PT_Processor):
    """
        Processor for DINO Tuning.
    """
    def __init__(self, argv=None, local_rank=-1):
        super().__init__(argv, local_rank)
        if 'NTU' in self.arg.train_feeder_args['data_path']:
            num_training_data = 40091
        else:
            raise ValueError("Dataset is not supported")
        niter_per_epoch = num_training_data // self.arg.batch_size
        self.momentum_scheduler = self.cosine_scheduler(self.arg.momentum_teacher, 1, self.arg.num_epoch, niter_per_epoch)
    
    def load_optimizer(self):
        parameters = []
        for name, param in self.model.encoder_student.named_parameters():
            appended = False
            if self.arg.third_stage:
                if name.startswith('blocks'):
                    block_idx = int(name.split('.')[1])
                    if block_idx >= 4:
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

    def train(self, epoch):
        self.model.train()
        loader = self.data_loader['train']
        loss_value = []

        selected_teacher_out = {}
        selected_student_out = {}

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
            m = self.momentum_scheduler[it]
            out_student, out_teacher = self.model(all_data, m)
            loss, kl, entropy = self.loss(out_student, out_teacher, epoch - 1)

            if self.arg.koleo_weight > 0:
                n_globals = self.arg.train_feeder_args['n_globals']
                n_locals = self.arg.train_feeder_args['n_globals']
                n_views = n_globals + n_locals
                batch_size = all_data[0].shape[0]
                out_student_global = out_student[:batch_size*n_views]
                koleo_loss = self.arg.koleo_weight * sum(
                    self.koleo_loss(p) for p in out_student_global.chunk(n_views)
                )
                loss += koleo_loss

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
            self.iter_info['kl'] = kl.data.item()
            self.iter_info['entropy'] = entropy.data.item()
            loss_value.append(self.iter_info['loss'])
            if self.local_rank == 0:
                self.show_iter_info()
            self.meta_info['iter'] += 1

            if self.arg.draw_distribution and epoch % self.arg.draw_freq == 0:
                for selected_idx in self.arg.selected_indices:
                    if selected_idx in indices:
                        batch_idx = torch.argwhere(indices == selected_idx).item()
                        selected_student_out[selected_idx] = out_student[batch_idx]
                        selected_teacher_out[selected_idx] = out_teacher[batch_idx]
                    
        self.epoch_info['train_mean_loss']= np.mean(loss_value)
        if self.local_rank == 0:
            self.show_epoch_info()

        if self.local_rank == 0 and len(selected_student_out) == len(self.arg.selected_indices):
            self.save_output_bar_chart(selected_teacher_out, selected_student_out, epoch)

    
    def save_output_bar_chart(self, teacher_out_all, student_out_all, epoch):
        fig, ax = plt.subplots(len(teacher_out_all), figsize=(10, 10))
        fig.suptitle(f'Distribution of outputs (Epoch {epoch})')
        plt.subplots_adjust(hspace=0.6)
        x_axis = np.arange(self.arg.model_args['feature_dim'])

        for i, out_idx in enumerate(teacher_out_all):
            teacher_out, student_out = teacher_out_all[out_idx], student_out_all[out_idx]
            ax[i].bar(x_axis - 0.2, teacher_out.cpu().detach().numpy(), 0.4, label="Teacher")
            ax[i].bar(x_axis + 0.2, student_out.cpu().detach().numpy(), 0.4, label="Student")
            ax[i].set_title(f"Sample {out_idx}")
            ax[i].legend()

        os.makedirs(f'figs/{self.wandb_id}', exist_ok=True)
        plt.savefig(f'figs/{self.wandb_id}/bar_chart_epoch{epoch:03d}.png')
        plt.close()
        
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
                else:
                    feature = self.model.module.encoder_student(data)
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
                        feature = self.model.module.encoder_student(data)
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
        parser.add_argument('--step', type=int, default=[], nargs='+', help='the epoch where optimizer reduce the learning rate')
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
