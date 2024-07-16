import argparse
import os
import shutil
import numpy as np
import random
import wandb

# torch
import torch
import torch.nn as nn
import torch.optim as optim

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# torchlight
import torchlight
from torchlight import str2bool
from torchlight import DictAction
from torchlight import import_class

from .io import IO
import wandb
import subprocess

def init_seed(seed=1):
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Processor(IO):
    """
        Base Processor
    """

    def __init__(self, argv=None, local_rank=-1):
        self.local_rank = local_rank
        self.load_arg(argv)
        self.init_environment()

        self.global_step = 0

    def init_environment(self):
        super().init_environment()
        self.result = dict()
        self.iter_info = dict()
        self.epoch_info = dict()
        self.eval_info = dict()
        self.meta_info = dict(epoch=0, iter=0)
        dist.init_process_group(backend='nccl')

    def load_model(self):
        raise NotImplementedError("Should be implemented in subclass")

    def load_loss(self):
        raise NotImplementedError("Should be implemented in subclass")
    
    def load_weights(self):
        if self.arg.weights:
            self.model = self.io.load_weights(self.model, self.arg.weights,
                                              self.arg.ignore_weights, self.arg.rename_weights)
    def load_optimizer(self):
        raise NotImplementedError("Should be implemented in subclass")
    
    def update_from_ckpt(self):
        raise NotImplementedError("Should be implemented in subclass")
    
    def wandb_init(self, updated):
        if updated:
            wandb.init(id=self.wandb_id,
                       project="Contrastive Action Recognition",
                       resume="must",
                       settings=wandb.Settings(start_method='fork'))
        else:
            wandb.init(name=self.arg.wandb_name,
                       project="Contrastive Action Recognition",
                       group=self.arg.wandb_group,
                       settings=wandb.Settings(start_method='fork'))
            wandb.config.update(self.arg)
            self.wandb_id = wandb.run.id

    def load_data(self):
        self.data_loader = dict()

        if self.arg.train_feeder_args:
            train_feeder = import_class(self.arg.train_feeder)
            train_dataset = train_feeder(**self.arg.train_feeder_args)
            self.train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
            self.data_loader['train'] = torch.utils.data.DataLoader(
                dataset=train_dataset,
                batch_size=self.arg.batch_size,
                num_workers=self.arg.num_worker * torchlight.ngpu(
                    self.arg.device),
                drop_last=True,
                sampler=self.train_sampler)

        if self.arg.test_feeder_args:
            test_feeder = import_class(self.arg.test_feeder)
            test_dataset = test_feeder(**self.arg.test_feeder_args)
            self.test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
            self.data_loader['test'] = torch.utils.data.DataLoader(
                dataset=test_dataset,
                batch_size=self.arg.test_batch_size,
                num_workers=self.arg.num_worker * torchlight.ngpu(
                    self.arg.device),
                drop_last=True,
                sampler=self.test_sampler)

        if self.arg.mem_train_feeder_args:
            mem_train_feeder = import_class(self.arg.mem_train_feeder)
            mem_train_dataset = mem_train_feeder(**self.arg.mem_train_feeder_args)
            self.mem_train_sampler = torch.utils.data.distributed.DistributedSampler(mem_train_dataset)
            self.data_loader['mem_train'] = torch.utils.data.DataLoader(
                dataset=mem_train_dataset,
                batch_size=self.arg.batch_size,
                num_workers=self.arg.num_worker * torchlight.ngpu(
                    self.arg.device),
                drop_last=False,
                sampler=self.mem_train_sampler)

        if self.arg.mem_test_feeder_args:
            mem_test_feeder = import_class(self.arg.mem_test_feeder)
            mem_test_dataset = mem_test_feeder(**self.arg.mem_test_feeder_args)
            self.mem_test_sampler = torch.utils.data.distributed.DistributedSampler(mem_test_dataset)
            self.data_loader['mem_test'] = torch.utils.data.DataLoader(
                dataset=mem_test_dataset,
                batch_size=self.arg.test_batch_size,
                num_workers=self.arg.num_worker * torchlight.ngpu(
                    self.arg.device),
                drop_last=False,
                sampler=self.mem_test_sampler)

    def show_epoch_info(self):
        for k, v in self.epoch_info.items():
            self.io.print_log('\t{}: {}'.format(k, v))
        if self.arg.pavi_log:
            self.io.log('train', self.meta_info['iter'], self.epoch_info)

    def show_eval_info(self):
        for k, v in self.eval_info.items():
            self.io.print_log('\t{}: {}'.format(k, v))
        if self.arg.pavi_log:
            self.io.log('eval', self.meta_info['iter'], self.eval_info)

    def show_iter_info(self):
        if self.meta_info['iter'] % self.arg.log_interval == 0:
            info ='\tIter {} Done.'.format(self.meta_info['iter'])
            for k, v in self.iter_info.items():
                if isinstance(v, float):
                    info = info + ' | {}: {:.4f}'.format(k, v)
                else:
                    info = info + ' | {}: {}'.format(k, v)

            self.io.print_log(info)

            if self.arg.pavi_log:
                self.io.log('train', self.meta_info['iter'], self.iter_info)

    def train(self):
        raise NotImplementedError("It should be implemented in subclass")

    def test(self):
        raise NotImplementedError("It should be implemented in subclass")

    def print_networks(self, net, print_flag=False):
        self.io.print_log('---------- Networks initialized -------------')
        num_params = 0
        for param in net.parameters():
            num_params += param.numel()
        if print_flag:
            self.io.print_log(net)
        self.io.print_log('[Network] Total number of parameters : %.3f M' % (num_params / 1e6))
        self.io.print_log('-----------------------------------------------')

    def start(self):
        if self.local_rank == 0:
            # get the output of `git diff`
            diff_output = subprocess.check_output(['git', 'diff'])
            # print the output
            self.io.print_log(diff_output.decode('utf-8'))
        
        torch.cuda.set_device(self.local_rank)
        self.device = torch.cuda.device(self.local_rank)
        self.dev = self.local_rank
        self.gpus = list(range(torch.distributed.get_world_size()))
        
        if self.arg.phase == 'train' and self.local_rank == 0:
            if os.path.isdir(self.arg.work_dir + '/train'):
                print('[INFO] log_dir: ', self.arg.work_dir, 'already exist')
                shutil.rmtree(self.arg.work_dir + '/train')
                shutil.rmtree(self.arg.work_dir + '/val')
                print('[INFO] Dir removed: ', self.arg.work_dir + '/train')
                print('[INFO] Dir removed: ', self.arg.work_dir + '/val')
        
        self.best_knn = 0 
        self.best_acc = 0.0
        self.wandb_id = wandb.util.generate_id()

        self.load_model() 
        self.load_loss()
        self.load_data()
        self.load_optimizer() 
        updated = self.update_from_ckpt()
        if self.local_rank == 0 and self.arg.use_wandb:
            self.wandb_init(updated)
            
        if self.local_rank == 0:    
            self.io.print_log('[HYPERPARAM] Parameters:')
            args_dict = vars(self.arg)
            for key, value in args_dict.items():
                if isinstance(value, dict):
                    self.io.print_log(f'[HYPERPARAM]\t{key}:')
                    for subkey in value:
                        self.io.print_log(f'[HYPERPARAM]\t\t {subkey}: {value[subkey]}')
                else:
                    self.io.print_log(f'[HYPERPARAM]\t{key}: {value}')
            self.print_networks(self.model)

        self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
        
        self.model = self.model.cuda()
        
        self.model = DDP(
                self.model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=False
            )

        # training phase
        if self.arg.phase == 'train':
            self.global_step = self.arg.start_epoch * len(self.data_loader['train'])
            self.meta_info['iter'] = self.global_step
            self.knn_results = dict()
            self.KNN_epoch_results = dict()
            for k in self.arg.knn_k:
                self.knn_results[k] = dict()
            
            for epoch in range(self.arg.start_epoch, self.arg.num_epoch):
                self.train_sampler.set_epoch(epoch)
                self.meta_info['epoch'] = epoch + 1

                # training
                if self.local_rank == 0:   
                    self.io.print_log('[INFO] Training epoch: {}'.format(epoch + 1))
                self.train(epoch + 1)

                # save model
                if self.arg.save_interval == -1:
                    pass
                elif (((epoch + 1) % self.arg.save_interval == 0) or (
                        epoch + 1 == self.arg.num_epoch)) and dist.get_rank() == 0:
                    self.io.save_model(self.model, name='last_epoch.pt')
                    self.io.save_pkl(result={
                                            'epoch': epoch,
                                            'optimizer': self.optimizer.state_dict(),
                                            'best_knn': self.best_knn,
                                            'best_acc': self.best_acc,
                                            'wandb_id': self.wandb_id,
                                            'loss': self.loss.state_dict() if hasattr(self, 'loss') else None
                                            }, 
                                     filename='last_epoch.pkl')

                # evaluation
                if self.arg.eval_interval == -1:
                    pass
                elif ((epoch + 1) % self.arg.eval_interval == 0) or (
                        epoch + 1 == self.arg.num_epoch):
                    if self.local_rank == 0:   
                        self.io.print_log('[INFO] Eval epoch: {}'.format(epoch + 1))
                    self.test(epoch + 1)
                    if self.local_rank == 0:   
                        self.io.print_log("[INFO] current %.2f%%, best %.2f%%" % 
                                                (self.current_acc, self.best_acc))

                    # save best model
                    if self.current_acc >= self.best_acc:  # current_acc is defined inside self.test -> self.show_best
                        filename = 'best_epoch.pt'
                        self.io.save_model(self.model, filename)
                        # save the output of model
                        if self.arg.save_result:
                            result_dict = dict(
                                zip(self.data_loader['test'].dataset.sample_name,
                                    self.result))
                            self.io.save_pkl(result_dict, 'test_result_best.pkl')

                if self.arg.knn_interval == -1:
                    pass
                elif ((epoch + 1) % self.arg.knn_interval == 0) or (epoch + 1 == self.arg.num_epoch):
                    if self.local_rank == 0:   
                        self.io.print_log('[INFO] KNN eval epoch {}'.format(epoch + 1))
                    self.knn_monitor(epoch + 1)

                    for k in self.arg.knn_k:
                        if self.local_rank == 0:
                            self.io.print_log("[INFO] KNN - {} current: {:.2f}%, best: {:.2f}%".format(
                                k, self.knn_results[k][epoch + 1], max(self.knn_results[k].values())
                                ))
                        if epoch + 1 in self.arg.KNN_show:
                            if epoch + 1 not in self.KNN_epoch_results:
                                self.KNN_epoch_results[epoch + 1] = dict()
                            self.KNN_epoch_results[epoch + 1][k] = [self.knn_results[k][epoch + 1], max(self.knn_results[k].values())]
                    
                    max_knn = self.get_max_knn()
                    if max_knn > self.best_knn:
                        self.best_knn = max_knn
                        if dist.get_rank() == 0:
                            self.io.save_model(self.model, name='best_epoch.pt')

                    if (epoch + 1 == self.arg.num_epoch) and self.local_rank == 0:
                        self.io.print_log('*' * 10 + ' KNN Result ' + '*' * 10)
                        for show_epoch in self.arg.KNN_show:
                            if show_epoch in self.KNN_epoch_results:
                                for k in self.arg.knn_k:
                                    self.io.print_log('\t{}-{}: cur-{:.2f}%, best-{:.2f}%'.format(
                                        show_epoch, k, self.KNN_epoch_results[show_epoch][k][0],
                                        self.KNN_epoch_results[show_epoch][k][1]))
                if self.local_rank == 0 and self.arg.use_wandb:    
                    self.log_to_wandb(epoch + 1)
            
            if self.local_rank == 0 and self.arg.use_wandb:    
                    artifact = wandb.Artifact('model', type='model')
                    artifact.add_file(os.path.join(self.arg.work_dir, 'last_epoch.pt'))
                    artifact.add_file(os.path.join(self.arg.work_dir, 'best_epoch.pt'))
                    wandb.log_artifact(artifact)


        # test phase
        elif self.arg.phase == 'test':
            
            # the path of weights must be appointed
            if not updated:
                raise ValueError('weights file after tuning is required.')
            self.io.print_log('Model:   {}.'.format(self.arg.model))
            self.io.print_log('Weights: {}.'.format(self.arg.weights))

            self.best_acc = 0.0

            # evaluation
            self.io.print_log('Evaluation Start:')
            self.test(1)
            self.io.print_log('Done.\n')

            # save the output of model
            if self.arg.save_result:
                result_dict = dict(
                    zip(self.data_loader['test'].dataset.sample_name,
                        self.result))
                self.io.save_pkl(result_dict, 'test_result_%.3d.pkl'% (epoch + 1))

    def log_to_wandb(self, epoch):
        log_dict = {}
        
        for key in self.iter_info:
            log_dict[key] = self.iter_info[key]

        if self.arg.knn_interval != -1 and ((epoch + 1) % self.arg.knn_interval == 0) or (epoch + 1 == self.arg.num_epoch):
            for k in self.arg.knn_k:
                log_dict[f'KNN/{k}-nn'] = self.knn_results[k][epoch]
        if self.arg.eval_interval != -1 and (((epoch + 1) % self.arg.eval_interval == 0) or (epoch + 1 == self.arg.num_epoch)):
            log_dict['eval/best_accuracy'] = self.best_acc
            log_dict['eval/accuracy'] = self.current_acc

        if os.path.exists(f'figs/{self.wandb_id}/bar_chart_epoch{epoch:03d}.png'):
            log_dict['bar_chart'] = wandb.Image(f'figs/{self.wandb_id}/bar_chart_epoch{epoch:03d}.png')
        if os.path.exists(f'figs/{self.wandb_id}/recon_epoch{epoch:03d}.gif'):
            log_dict['reconstruction'] = wandb.Image(f'figs/{self.wandb_id}/recon_epoch{epoch:03d}.gif')
        wandb.log(log_dict)

    def get_max_knn(self):
        max_knn = 0
        for k in self.knn_results:
            max_knn_k = max(self.knn_results[k].values())
            if max_knn_k > max_knn:
                max_knn = max_knn_k
        return max_knn

    @staticmethod
    def get_parser(add_help=False):

        # region arguments yapf: disable
        # parameter priority: command line > config > default
        parser = argparse.ArgumentParser(add_help=add_help, description='Base Processor')

        parser.add_argument('-w', '--work_dir', default='./work_dir/tmp', help='the work folder for storing results')
        parser.add_argument('-c', '--config', default=None, help='path to the configuration file')

        # processor
        parser.add_argument('--phase', default='train', help='must be train or test')
        parser.add_argument('--save_result', type=str2bool, default=True,
                            help='if ture, the output of the model will be stored')
        parser.add_argument('--start_epoch', type=int, default=0, help='start training from which epoch')
        parser.add_argument('--num_epoch', type=int, default=80, help='stop training in which epoch')
        parser.add_argument('--use_gpu', type=str2bool, default=True, help='use GPUs or not')
        parser.add_argument('--device', type=int, default=0, nargs='+',
                            help='the indexes of GPUs for training or testing')

        # visulize and debug
        parser.add_argument('--log_interval', type=int, default=100,
                            help='the interval for printing messages (#iteration)')
        parser.add_argument('--save_interval', type=int, default=10, help='the interval for storing models (#epoch)')
        parser.add_argument('--eval_interval', type=int, default=5, help='the interval for evaluating models (#epoch)')
        parser.add_argument('--save_log', type=str2bool, default=True, help='save logging or not')
        parser.add_argument('--print_log', type=str2bool, default=True, help='print logging or not')
        parser.add_argument('--pavi_log', type=str2bool, default=False, help='logging on pavi or not')

        parser.add_argument('--knn_interval', type=int, default=5, help='the interval for knn models (#epoch)')

        # feeder
        parser.add_argument('--train_feeder', default='feeder.feeder', help='train data loader will be used')
        parser.add_argument('--test_feeder', default='feeder.feeder', help='test data loader will be used')
        parser.add_argument('--num_worker', type=int, default=4, help='the number of worker per gpu for data loader')
        parser.add_argument('--train_feeder_args', action=DictAction, default=dict(),
                            help='the arguments of data loader for training')
        parser.add_argument('--test_feeder_args', action=DictAction, default=dict(),
                            help='the arguments of data loader for test')
        parser.add_argument('--mem_train_feeder', default='feeder.feeder',
                            help='memory train data loader will be used for knn')
        parser.add_argument('--mem_test_feeder', default='feeder.feeder',
                            help='memory test data loader will be used for knn')
        parser.add_argument('--mem_train_feeder_args', action=DictAction, default=dict(),
                            help='the arguments of data loader for training')
        parser.add_argument('--mem_test_feeder_args', action=DictAction, default=dict(),
                            help='the arguments of data loader for test')

        parser.add_argument('--batch_size', type=int, default=256, help='training batch size')
        parser.add_argument('--test_batch_size', type=int, default=256, help='test batch size')
        parser.add_argument('--debug', action="store_true", help='less data, faster loading')

        # model
        parser.add_argument('--model', default=None, help='the model will be used')
        parser.add_argument('--model_args', action=DictAction, default=dict(), help='the arguments of model')
        parser.add_argument('--weights', default=None, help='the weights for network initialization')
        parser.add_argument('--ignore_weights', type=str, default=[], nargs='+', help='the name of weights which will be ignored in the initialization')
        parser.add_argument('--rename_weights', type=str, default=[], nargs='+', help='the name of weights which will be ignored in the initialization')
                            
        parser.add_argument('--knn_k', type=int, default=[], nargs='+', help='KNN-K')
        parser.add_argument('--knn_classes', type=int, default=60, help='use cosine lr schedule')
        parser.add_argument('--knn_t', type=float, default=0.1, help='use cosine lr schedule')
        parser.add_argument('--KNN_show', type=int, default=[], nargs='+',
                            help='the epoch to show the best KNN result')
        # endregion yapf: enable

        # wandb
        parser.add_argument('--use-wandb', action="store_true", help='Using wandb for logging purpose')
        parser.add_argument('--wandb-name', type=str, default=None, help="wandb run name")
        parser.add_argument('--wandb-group', type=str, default=None, help="wandb run name")
        parser.add_argument('--wandb-id', type=str, default=None, help="wandb id (It's not used usually.)")

        return parser
