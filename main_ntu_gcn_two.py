#!/usr/bin/env python
from __future__ import print_function

import argparse

import os
import pickle
import random

import sys
import time
import pprint
from collections import OrderedDict, defaultdict
import traceback
from torch.optim.lr_scheduler import MultiStepLR
import torch.nn.functional as F
import numpy as np

# torch
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import yaml
from tqdm import tqdm
import apex



pairs = {
    'ntu': (
        (1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6), (8, 7), (9, 21), (10, 9), (11, 10), (12, 11),
        (13, 1),
        (14, 13), (15, 14), (16, 15), (17, 1), (18, 17), (19, 18), (20, 19), (22, 23), (21, 21), (23, 8), (24, 25),
        (25, 12)
    )}


def init_seed(seed):
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def import_class(import_str):
    mod_str, _sep, class_str = import_str.rpartition('.')
    __import__(mod_str)
    try:
        return getattr(sys.modules[mod_str], class_str)
    except AttributeError:
        raise ImportError('Class %s cannot be found (%s)' % (class_str, traceback.format_exception(*sys.exc_info())))


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


class DictAction(argparse.Action):
    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        if nargs is not None:
            raise ValueError("nargs not allowed")
        super(DictAction, self).__init__(option_strings, dest, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        input_dict = eval(f'dict({values})')  # pylint: disable=W0123
        output_dict = getattr(namespace, self.dest)
        for k in input_dict:
            output_dict[k] = input_dict[k]
        setattr(namespace, self.dest, output_dict)


def get_parser():
    # parameter priority: command line > config > default
    parser = argparse.ArgumentParser(
        description='Spatial Temporal Graph Convolution Network For HAR')

    parser.add_argument(
        '--work-dir',
        help='the work folder for storing results')

    parser.add_argument(
        '--config',
        default='config/ntu_xsub/train_gcn_two.yaml',
        help='path to the configuration file')

    # processor
    parser.add_argument(
        '--phase', default='train', help='must be train or test')

    parser.add_argument(
        '--save-score',
        type=str2bool,
        default=True,
        help='if ture, the classification score will be stored')

    # visulize and debug
    parser.add_argument(
        '--seed', type=int, default=1, help='random seed for pytorch')



    parser.add_argument(
        '--print-log',
        type=str2bool,
        default=True,
        help='print logging or not')

    parser.add_argument(
        '--show-topk',
        type=int,
        default=[1, 5],
        nargs='+',
        help='which Top K accuracy will be shown')

    # feeder
    parser.add_argument(
        '--feeder', default='feeder.feeder', help='data loader will be used')

    parser.add_argument(
        '--train-feeder-args',
        action=DictAction,
        default=dict(),
        help='the arguments of data loader for training')
    parser.add_argument(
        '--test-feeder-args',
        action=DictAction,
        default=dict(),
        help='the arguments of data loader for test')

    # model
    parser.add_argument('--model', default=None, help='the model will be used')

    parser.add_argument(
        '--weights1',
        default=None,
        help='the weights for network initialization')
    parser.add_argument(
        '--weights2',
        default=None,
        help='the weights for network initialization')

    # optim
    parser.add_argument(
        '--base-lr', type=float, default=0.001, help='initial learning rate')
    parser.add_argument(
        '--step',
        type=int,
        default=[20, 40, 60],
        nargs='+',
        help='the epoch where optimizer reduce the learning rate')
    parser.add_argument(
        '--device',
        type=int,
        default=0,
        nargs='+',
        help='the indexes of GPUs for training or testing')

    parser.add_argument(
        '--flag',
        type=int,
        default=1)

    # nesterov 牛顿运算
    parser.add_argument(
        '--is-test', type=str2bool, default=False, help='use nesterov or not')

    parser.add_argument('--optimizer', default='SGD', help='type of optimizer')

    # nesterov 牛顿运算
    parser.add_argument(
        '--nesterov', type=str2bool, default=False, help='use nesterov or not')


    # 最小切片
    parser.add_argument(
        '--forward-batch-size', type=int, default=256, help='mini batch size')

    parser.add_argument(
        '--batch-size', type=int, default=256, help='training batch size')
    parser.add_argument(
        '--test-batch-size', type=int, default=256, help='test batch size')


    parser.add_argument(
        '--num-epoch',
        type=int,
        default=80,
        help='stop training in which epoch')
    parser.add_argument(
        '--weight-decay',
        type=float,
        default=0.0005,
        help='weight decay for optimizer')
    # is lr-decay
    parser.add_argument(
        '--lr-decay', type=str2bool, default=False, help='use lr-decay or not')

    parser.add_argument(
        '--lr-decay-rate',
        type=float,
        default=0.1,
        help='decay rate for learning rate')


    parser.add_argument(
        '--half', type=str2bool, default=False, help='use nesterov or not')
    parser.add_argument('--amp_opt_level', type=int, default=1)

    return parser


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class Processor():
    """Processor for Skeleton-based Action Recgnition"""

    def __init__(self, arg):
        self.arg = arg

        if not os.path.exists(arg.work_dir):
            os.makedirs(arg.work_dir)
        self.classes = 60

        self.load_model()
        self.load_param_groups()
        self.load_optimizer()
        self.load_lr_scheduler()
        self.load_data()

        self.lr = self.arg.base_lr

        self.base_epoch = 0

        self.best_acc = 0
        self.best_acc_epoch = 0

        self.train_acc = []
        self.train_loss = []

        self.test_acc = []
        self.test_loss = []

        if self.arg.half:
            self.print_log('*************************************')
            self.print_log('*** Using Half Precision Training ***')
            self.print_log('*************************************')
            self.model, self.optimizer = apex.amp.initialize(
                self.model,
                self.optimizer,
                opt_level=f'O{1}'
            )
            if self.arg.amp_opt_level != 1:
                self.print_log('[WARN] nn.DataParallel is not yet supported by amp_opt_level != "O1"')



    def load_model(self):

        output_device = self.arg.device[0] if type(
            self.arg.device) is list else self.arg.device

        self.output_device = output_device

        # import model and loss
        Model = import_class(self.arg.model)

        self.model = Model().cuda(output_device)



        self.print_log(f'Model1 total number of params: {count_params(self.model)}')


        if self.arg.weights1:
            self.print_log(f'Loading weights from {self.arg.weights1}')

            weights = torch.load(self.arg.weights1)
            weights = OrderedDict(
                [[k.split('module.')[-1],
                  v.cuda(output_device)] for k, v in weights.items()])
            try:
                self.model.SingleModel1.load_state_dict(weights)
            except:
                state = self.model.SingleModel1.state_dict()
                diff = list(set(state.keys()).difference(set(weights.keys())))
                self.print_log('Can not find these weights:')
                for d in diff:
                    self.print_log('  ' + d)
                state.update(weights)
                self.model.SingleModel1.load_state_dict(state)
        if self.arg.weights2:
            self.print_log(f'Loading weights from {self.arg.weights2}')

            weights = torch.load(self.arg.weights2)
            weights = OrderedDict(
                [[k.split('module.')[-1],
                  v.cuda(output_device)] for k, v in weights.items()])
            try:
                self.model.SingleModel2.load_state_dict(weights)
            except:
                state = self.model.SingleModel2.state_dict()
                diff = list(set(state.keys()).difference(set(weights.keys())))
                self.print_log('Can not find these weights:')
                for d in diff:
                    self.print_log('  ' + d)
                state.update(weights)
                self.model.SingleModel2.load_state_dict(state)


    def load_param_groups(self):
        """
        Template function for setting different learning behaviour
        (e.g. LR, weight decay) of different groups of parameters
        """
        self.param_groups = defaultdict(list)

        for name, params in self.model.named_parameters():
            self.param_groups['other'].append(params)

        self.optim_param_groups = {
            'other': {'params': self.param_groups['other']}
        }

    def load_optimizer(self):
        params = list(self.optim_param_groups.values())
        if self.arg.optimizer == 'SGD':
            self.optimizer = optim.SGD(
                params,
                lr=self.arg.base_lr,
                momentum=0.9,
                nesterov=self.arg.nesterov,
                weight_decay=self.arg.weight_decay)
        elif self.arg.optimizer == 'Adam':
            self.optimizer = optim.Adam(
                params,
                lr=self.arg.base_lr,
                weight_decay=self.arg.weight_decay)
        else:
            raise ValueError('Unsupported optimizer: {}'.format(self.arg.optimizer))



    def load_lr_scheduler(self):
        self.lr_scheduler = MultiStepLR(self.optimizer, milestones=self.arg.step, gamma=0.1)

    def load_data(self):
        Feeder = import_class(self.arg.feeder)
        self.data_loader = dict()
        def worker_seed_fn(worker_id):
            return init_seed(self.arg.seed + worker_id + 1)
        self.data_loader['train'] = torch.utils.data.DataLoader(
                dataset=Feeder(**self.arg.train_feeder_args),
                batch_size=self.arg.batch_size,
                shuffle=True,
                drop_last=True,
                worker_init_fn=worker_seed_fn)

        self.data_loader['test'] = torch.utils.data.DataLoader(
            dataset=Feeder(**self.arg.test_feeder_args),
            batch_size=self.arg.test_batch_size,
            shuffle=False,
            drop_last=False,
            worker_init_fn=worker_seed_fn)

    def print_log(self, s, print_time=True):
        if print_time:
            localtime = time.asctime(time.localtime(time.time()))
            s = f'[ {localtime} ] {s}'
        print(s)
        if self.arg.print_log:
            with open(os.path.join(self.arg.work_dir, 'log.txt'), 'a') as f:
                print(s, file=f)

    def print_time(self):
        localtime = time.asctime(time.localtime(time.time()))
        self.print_log(f'Local current time: {localtime}')

    def record_time(self):
        self.cur_time = time.time()
        return self.cur_time

    def split_time(self):
        split_time = time.time() - self.cur_time
        self.record_time()
        return split_time

    def save_states(self, states, out_folder, out_name):
        out_folder_path = os.path.join(self.arg.work_dir, out_folder)
        out_path = os.path.join(out_folder_path, out_name)
        os.makedirs(out_folder_path, exist_ok=True)
        torch.save(states, out_path)

    def DS_Combin(self, alpha):
        def DS_Combin_two(alpha1, alpha2):
            alpha = dict()
            alpha[0], alpha[1] = alpha1, alpha2
            b, S, E, u = dict(), dict(), dict(), dict()
            for v in range(2):
                S[v] = torch.sum(alpha[v], dim=1, keepdim=True)
                E[v] = alpha[v] - 1
                b[v] = E[v] / (S[v].expand(E[v].shape))
                u[v] = self.classes / S[v]

            # b^0 @ b^(0+1)
            bb = torch.bmm(b[0].view(-1, self.classes, 1), b[1].view(-1, 1, self.classes))
            # b^0 * u^1
            uv1_expand = u[1].expand(b[0].shape)
            bu = torch.mul(b[0], uv1_expand)
            # b^1 * u^0
            uv_expand = u[0].expand(b[0].shape)
            ub = torch.mul(b[1], uv_expand)
            # calculate C
            bb_sum = torch.sum(bb, dim=(1, 2), out=None)
            bb_diag = torch.diagonal(bb, dim1=-2, dim2=-1).sum(-1)
            C = bb_sum - bb_diag

            # calculate b^a
            b_a = (torch.mul(b[0], b[1]) + bu + ub) / ((1 - C).view(-1, 1).expand(b[0].shape))
            # calculate u^a
            u_a = torch.mul(u[0], u[1]) / ((1 - C).view(-1, 1).expand(u[0].shape))

            # calculate new S
            S_a = self.classes / u_a
            # calculate new e_k
            e_a = torch.mul(b_a, S_a.expand(b_a.shape))
            alpha_a = e_a + 1
            return alpha_a

        for v in range(len(alpha) - 1):
            if v == 0:
                alpha_a = DS_Combin_two(alpha[0], alpha[1])
            else:
                alpha_a = DS_Combin_two(alpha_a, alpha[v + 1])
        return alpha_a

    def bal(self, b_j, b_k):
        if b_j * b_k != 0:
            return 1 - torch.abs(b_j - b_k) / (b_j + b_k)
        else:
            return 0

    def diss(self, alpha):
        N, K = alpha.shape

        # 计算每个样本的 b 的总和和平方和
        sum_b = alpha.sum(dim=1, keepdim=True)
        sum_b_squared = alpha.pow(2).sum(dim=1, keepdim=True)

        # 计算分母
        denominator = sum_b - sum_b_squared / sum_b

        # 计算 Bal 矩阵
        alpha_expanded = alpha.unsqueeze(2)  # 扩展 alpha 以进行广播
        alpha_transpose = alpha_expanded.transpose(1, 2)

        # 计算 b_j 和 b_k 之间的差值和和
        diff = torch.abs(alpha_expanded - alpha_transpose)
        sum_bjk = alpha_expanded + alpha_transpose
        # 避免除以零
        sum_bjk = torch.where(sum_bjk != 0, sum_bjk, 1)

        # 计算 Bal 矩阵
        Bal = 1 - diff / sum_bjk
        # 计算分子
        product = alpha_expanded * Bal * alpha_transpose
        numerator = product.sum(dim=2)
        # 计算 diss(α) 对于每个样本
        diss_alpha = (numerator / denominator).sum(dim=1)

        return diss_alpha

    def get_alpha(self, e):
        e = F.softplus(e)
        alpha = e + 1

        S = torch.sum(alpha, dim=-1, keepdim=True)
        b = e / S
        d = self.diss(b)

        return alpha


    def MSE(self, alpha, label):
        num_classes = alpha.shape[-1]
        S = torch.sum(alpha, dim=-1, keepdim=True)
        label = F.one_hot(label, num_classes)
        p = alpha / S
        u = num_classes / S
        A = torch.sum((label - p) ** 2, dim=-1, keepdim=True)
        B = torch.sum(alpha * (S - alpha) / (S * S * (S + 1)), dim=-1, keepdim=True)
        return torch.mean(A + B), p, u

    def train(self, epoch):
        self.model.train()

        loader = self.data_loader['train']

        loss_values = []
        epoch_acc = []
        self.print_log('Now epoch:{}'.format(epoch + 1))
        self.record_time()

        timer = dict(dataloader=0.001, model=0.001, statistics=0.001)
        # 是否更新学习率
        current_lr = self.optimizer.param_groups[0]['lr']

        self.print_log(f'Training is staring. Training epoch: {epoch + 1}, LR: {current_lr:.4f}')

        process = tqdm(loader, dynamic_ncols=True)
        for batch_idx, (data, label) in enumerate(process):
            # get data
            data = data.float().cuda(self.output_device)
            label = label.long().cuda(self.output_device)
            timer['dataloader'] += self.split_time()

            # backward
            self.optimizer.zero_grad()

            ############## Gradient Accumulation for Smaller Batches ##############
            real_batch_size = self.arg.forward_batch_size
            splits = len(data) // real_batch_size
            assert len(data) % real_batch_size == 0, \
                'Real batch size should be a factor of arg.batch_size!'

            for i in range(splits):

                left = i * real_batch_size
                right = left + real_batch_size
                batch_joint, batch_label = data[left:right], label[left:right]


                batch_tmp = batch_joint.clone()
                batch_bone = torch.zeros_like(batch_tmp)
                for v1, v2 in pairs['ntu']:
                    batch_bone[:, :,:, v1-1, :] = batch_tmp[:, :,:, v1-1, :] - batch_tmp[:, :,:, v2-1, :]


                batch_joint = batch_joint.float().cuda(self.output_device)
                batch_bone = batch_bone.float().cuda(self.output_device)

                # forward
                out1,out2,out3 = self.model(batch_joint,batch_bone)
                alpha1 = self.get_alpha(out1)
                alpha2 = self.get_alpha(out1)
                alpha3 = self.get_alpha(out1)

                output = out1
                loss,p,u = self.MSE(output,batch_label)

                if self.arg.half:
                    with apex.amp.scale_loss(loss, self.optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                loss_values.append(loss.item())

                timer['model'] += self.split_time()

                # Display loss
                process.set_description(f'(BS {real_batch_size}) loss: {loss.item():.4f}')

                value, predict_label = torch.max(output, 1)
                acc = torch.mean((predict_label == batch_label).float())

                epoch_acc.append(acc)

            #####################################

            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 2)
            self.optimizer.step()

            # statistics
            self.lr = self.optimizer.param_groups[0]['lr']

            timer['statistics'] += self.split_time()

            # Delete output/loss after each batch since it may introduce extra mem during scoping
            # https://discuss.pytorch.org/t/gpu-memory-consumption-increases-while-training/2770/3
            del output
            del loss

        # statistics of time consumption and loss
        proportion = {
            k: f'{int(round(v * 100 / sum(timer.values()))):02d}%'
            for k, v in timer.items()
        }

        mean_loss = np.mean(loss_values)
        mean_acc = np.mean(np.array([item.cpu().numpy() for item in epoch_acc]))
        num_splits = self.arg.batch_size // self.arg.forward_batch_size

        self.print_log(f'\tMean training loss: {mean_loss:.4f} (num_splits {num_splits}).')
        self.print_log(
            f'\tMean training acc: {mean_acc * 100:.2f}%.')
        self.print_log('\tTime consumption: [Data]{dataloader}, [Network]{model}'.format(**proportion))

        self.lr_scheduler.step()

        self.train_acc.append(mean_acc)
        self.train_loss.append(mean_loss)

    def eval(self, epoch):
        with torch.no_grad():
            self.model = self.model.cuda(self.output_device)
            self.model.eval()
            self.print_log(f'Eval is staring. Eval epoch: {epoch + 1}')

            loss_values = []
            score_batches = []
            ln = 'test'

            process = tqdm(self.data_loader['test'], dynamic_ncols=True)
            for batch_idx, (joint, label) in enumerate(process):
                batch_tmp = joint.clone()
                bone = torch.zeros_like(batch_tmp)
                for v1, v2 in pairs['ntu']:
                    bone[:, :, :, v1 - 1, :] = batch_tmp[:, :, :, v1 - 1, :] - batch_tmp[:, :, :, v2 - 1, :]

                label = label.long().cuda(self.output_device)
                joint = joint.float().cuda(self.output_device)
                bone = bone.float().cuda(self.output_device)

                # forward
                out1, out2, out3 = self.model(joint, bone)
                alpha1 = self.get_alpha(out1)
                alpha2 = self.get_alpha(out1)
                alpha3 = self.get_alpha(out1)

                output = out1
                loss, p, u = self.MSE(output, label)


                score_batches.append(output.data.cpu().numpy())
                loss_values.append(loss.item())

                _, predict_label = torch.max(output.data, 1)


            score = np.concatenate(score_batches)
            loss = np.mean(loss_values)
            accuracy = self.data_loader['test'].dataset.top_k(score, 1)

            self.test_acc.append(accuracy)
            self.test_loss.append(loss)

            self.print_log(
                f'\t work_dir: {self.arg.work_dir}.')
            self.print_log(
                f'\tMean test acc: {accuracy * 100:.2f}%.')
            score_dict = dict(zip(self.data_loader[ln].dataset.sample_name, score))
            self.print_log(f'\tMean {ln} loss of {len(self.data_loader[ln])} batches: {np.mean(loss_values)}.')


            for k in self.arg.show_topk:
                self.print_log(f'\tTop {k}: {100 * self.data_loader[ln].dataset.top_k(score, k):.2f}%')

            if accuracy > self.best_acc:
                self.print_log(f'Last epoch to save weight and checkpoint.!!!!!!! !!!!!!!!!!!!!!!!!!!!!!!!!!!! Epoch number is {epoch + 1}')
                self.best_acc = accuracy
                self.best_acc_epoch = epoch + 1

                weight_state_dict = self.model.state_dict()
                weights = OrderedDict([
                    [k.split('module.')[-1], v.cpu()]
                    for k, v in weight_state_dict.items()
                ])

                weights_name = f'best_acc_weights.pt'
                self.save_states(weights, './', weights_name)


                with open('{}/best_acc_result.pkl'.format(self.arg.work_dir), 'wb') as f:
                    pickle.dump(score_dict, f)

        # Empty cache after evaluation
        torch.cuda.empty_cache()

    def start(self):

            self.print_log(f'Parameters:\n{pprint.pformat(vars(self.arg))}\n')
            self.print_log(f'Model total number of params: {count_params(self.model)}')
            if self.arg.is_test:
                self.eval(0)
            else:
                for epoch in range(0, self.arg.num_epoch):
                    # self.train(epoch)
                    self.eval(epoch)

                num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
                self.print_log(f'Best accuracy: {self.best_acc}')
                self.print_log(f'Epoch number: {self.best_acc_epoch}')
                self.print_log(f'Model work_dir: {self.arg.work_dir}')
                self.print_log(f'Model total number of params: {num_params}')
                self.print_log(f'Weight decay: {self.arg.weight_decay}')
                self.print_log(f'Base LR: {self.arg.base_lr}')
                self.print_log(f'Batch Size: {self.arg.batch_size}')
                self.print_log(f'Forward Batch Size: {self.arg.forward_batch_size}')
                self.print_log(f'Test Batch Size: {self.arg.test_batch_size}')

                self.print_log('Done.\n')


def main():
    parser = get_parser()
    # load arg form config file
    p = parser.parse_args()
    if p.config is not None:
        with open(p.config, 'r',encoding='utf-8') as f:
            default_arg = yaml.load(f, Loader=yaml.FullLoader)
        key = vars(p).keys()
        for k in default_arg.keys():
            if k not in key:
                print('WRONG ARG:', k)
                assert (k in key)
        parser.set_defaults(**default_arg)

    arg = parser.parse_args()
    init_seed(arg.seed)

    
    processor = Processor(arg)
    processor.start()


if __name__ == '__main__':
    main()

