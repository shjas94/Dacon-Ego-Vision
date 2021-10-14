# https://github.com/facebookresearch/mixup-cifar10
# https://github.com/Bjarten/early-stopping-pytorch


import random
import os
import glob
import pandas as pd
import numpy as np
import torch
from modules.optimizers import *
from modules.losses import *


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def get_optimizer(cfg, model):
    if cfg['TRAIN']['OPTIMIZER'] == 'radam':
        return RAdam(params=model.parameters(), lr=float(cfg['TRAIN']['LR']), weight_decay=float(cfg['TRAIN']['WEIGHT_DECAY']))
    elif cfg['TRAIN']['OPTIMIZER'] == 'adamw':
        return AdamW(params=model.parameters(), lr=float(cfg['TRAIN']['LR']), weight_decay=float(cfg['TRAIN']['WEIGHT_DECAY']))
    elif cfg['TRAIN']['OPTIMIZER'] == 'sam':
        return SAM(model.parameters(), RAdam, lr=float(cfg['TRAIN']['LR']), weight_decay=float(cfg['TRAIN']['WEIGHT_DECAY']))
    else:
        return torch.optim.Adam(params=model.parameters(), lr=float(cfg['TRAIN']['LR']), weight_decay=float(cfg['TRAIN']['WEIGHT_DECAY']))


def get_criterion(cfg_criterion, cfg):
    if cfg_criterion == 'ce':
        return torch.nn.CrossEntropyLoss()
    elif cfg_criterion == 'focal':
        return FocalLoss(gamma=cfg['TRAIN']['GAMMA'])


def collate_fn(batch):
    return tuple(zip(*batch))


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(
                f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(
                f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


def get_test_paths(cfg):
    test_dirs = sorted(glob.glob(os.path.join(
        cfg['PATH']['DATA'], 'test', '*')), key=lambda x: int(x.split('/')[4]))
    paths = [t for t in test_dirs]
    test_dirs = pd.DataFrame({'path': paths})
    return test_dirs


def softmax(x):
    # returns max of each row and keeps same dims
    max = np.max(x, axis=1, keepdims=True)
    e_x = np.exp(x - max)  # subtracts each row with its max value
    # returns sum of each row and keeps same dims
    sum = np.sum(e_x, axis=1, keepdims=True)
    f_x = e_x / sum
    return f_x
