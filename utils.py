import os
import random
import numpy as np

import torch


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)  # cpu vars
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def distributed(model, device, n_gpu):
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)
        model = model.to(device)
    else:
        assert n_gpu == 1
        model = model.to(device)
    return model


def count_parameters(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params / 1000000
