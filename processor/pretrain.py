import argparse
import os
import collections
from tqdm import tqdm
import pickle
import numpy as np

# torch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# torchlight
from torchlight import str2bool

from .processor import Processor

from .knn_monitor import knn_predict
from net.generic import concat_all_gather
from loss.dino import DINOLoss
from loss.reconstruction import loss_mpjpe, sce_loss
from loss.regularizer import KoLeoLoss

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv1d') != -1 or classname.find('Conv2d') != -1 or classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

        
class PT_Processor(Processor):
    """
        Processor for Pretraining.
    """

    def load_pretrained_weights(self, model, checkpoint):
        """Load pretrianed weights to model
        Incompatible layers (unmatched in name or size) will be ignored
        Args:
        - model (nn.Module): network model, which must not be nn.DataParallel
        - weight_path (str): path to pretrained weights
        """
        state_dict = checkpoint
        model_dict = model.state_dict()
        new_state_dict = collections.OrderedDict()
        matched_layers, discarded_layers = [], []
        for k, v in state_dict.items():
            # If the pretrained state_dict was saved as nn.DataParallel,
            # keys would contain "module.", which should be ignored.
            if k.startswith('module.'):
                k = k[7:]
            if k in model_dict and model_dict[k].size() == v.size():
                new_state_dict[k] = v
                matched_layers.append(k)
            else:
                discarded_layers.append(k)
        model_dict.update(new_state_dict)
        model.load_state_dict(model_dict, strict=True)
        self.io.print_log(f'(load_pretrained_weights) {len(matched_layers)} layers are loaded')
        discarded_layers = [key for key in discarded_layers if not (key.startswith('decoder') or key.startswith('mask'))]
        self.io.print_log(f'(load_pretrained_weights) {len(discarded_layers)} layers are discared')
        if len(discarded_layers) != 0:
            exit(1)
        return model

    def load_model(self):
        self.model = self.io.load_model(self.arg.model,
                                        **(self.arg.model_args))
        self.model.apply(weights_init)
        if 'weights' in self.arg.model_args:
            checkpoint = torch.load(self.arg.model_args['weights'])
            if 'model' in checkpoint:
                self.io.print_log("Loading weights for encoder_student")
                self.load_pretrained_weights(self.model.encoder_student, checkpoint['model'])
                if hasattr(self.model, 'encoder_teacher'):
                    self.io.print_log("Loading weights for encoder_teacher")
                    self.load_pretrained_weights(self.model.encoder_teacher, checkpoint['model'])
            else:
                self.io.print_log("Loading weights for model")
                self.model = self.io.load_weights(self.model, self.arg.model_args['weights'])
                self.io.print_log("Loaded")
    
    def load_loss(self):
        if self.arg.loss == "cross-entropy":
            self.loss = nn.CrossEntropyLoss()
        elif self.arg.loss == "dino":
            self.loss = DINOLoss(out_dim=self.arg.model_args['feature_dim'],
                                 n_locals=self.arg.train_feeder_args.get('n_locals', 0),
                                 n_globals=self.arg.train_feeder_args.get('n_globals', 2),
                                 warmup_teacher_temp=self.arg.warmup_teacher_temp,
                                 warmup_teacher_temp_epochs=self.arg.warmup_teacher_temp_epochs,
                                 teacher_temp=self.arg.teacher_temp,
                                 student_temp=self.arg.student_temp,
                                 nepochs=self.arg.num_epoch,
                                 batch_size=self.arg.batch_size,
                                 centering_type=self.arg.centering_type
                                 ).to(self.dev)
            self.loss_contrastive = nn.CrossEntropyLoss()
        elif self.arg.loss == '':
            pass
        else:
            raise NotImplementedError("Loss is not supported")
        
        if hasattr(self.arg, 'koleo_weight'):
            self.koleo_loss = KoLeoLoss()
        
    def load_optimizer(self):
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
        elif self.arg.optimizer == "AdamW":
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.arg.base_lr,
                betas=(0.9, 0.95)
            )
        else:
            raise ValueError()
        
    def update_from_ckpt(self):
        weights_path = os.path.join(self.arg.work_dir, 'last_epoch.pt')
        train_ckpt_path = os.path.join(self.arg.work_dir, 'last_epoch.pkl')
        if os.path.exists(weights_path):
            assert os.path.exists(train_ckpt_path), "Train checkpoint path does not exist."

            self.model = self.io.load_weights(self.model, weights_path,
                                              self.arg.ignore_weights, self.arg.rename_weights)
            self.model.to(self.dev)
            with open(train_ckpt_path, 'rb') as fp:
                train_ckpt = pickle.load(fp)
            self.arg.start_epoch = train_ckpt['epoch'] + 1
            self.optimizer.load_state_dict(train_ckpt['optimizer'])
            self.best_knn = train_ckpt['best_knn']
            self.wandb_id = train_ckpt.get('wandb_id', self.arg.wandb_id)
            if hasattr(self, 'loss'):
                self.loss.load_state_dict(train_ckpt['loss'])
            return True
        return False

    def adjust_lr(self):
        if self.arg.step:
            lr = self.arg.base_lr * (
                0.1**np.sum(self.meta_info['epoch'] > np.array(self.arg.step)))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            self.lr = lr
        else:
            self.lr = self.arg.base_lr

    def train(self, epoch):
        raise NotImplementedError("It should be implemented in subclass")

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
        parser.add_argument('--loss', default='cross-entropy', help='Loss function')
        return parser

    @torch.no_grad()
    def knn_monitor(self, epoch):
        self.model.module.encoder_q.eval()
        feature_bank, label_bank = [], []
        with torch.no_grad():
            # generate feature bank
            for data, label in tqdm(self.data_loader['mem_train'], desc='Feature extracting'):
                data = data.float().to(self.dev, non_blocking=True)
                label = label.long().to(self.dev, non_blocking=True)

                data = self.view_gen(data)
                feature = self.model.module.encoder_q(data)
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

                    feature = self.model.module.encoder_q(data)
                    feature = concat_all_gather(F.normalize(feature, dim=1))

                    pred_labels = knn_predict(feature, feature_bank, feature_labels, self.arg.knn_classes, i,
                                              self.arg.knn_t)

                    total_num += feature.size(0)
                    total_top1 += (pred_labels[:, 0] == concat_all_gather(label)).float().sum().item()
                    test_bar.set_postfix({'k': i, 'Accuracy': total_top1 / total_num * 100})
                acc = total_top1 / total_num * 100

                self.knn_results[i][epoch] = acc