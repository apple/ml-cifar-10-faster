#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2019 Apple Inc. All Rights Reserved.
#

import csv
import copy
import time
import inspect
import torchvision
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from itertools import chain
from inspect import signature
from functools import lru_cache as cache
from collections import defaultdict, namedtuple

####################
# Constants
####################

CIFAR10_MEAN, CIFAR10_STD = [
    (125.31, 122.95, 113.87),  # equals np.mean(cifar10()['train']['data'], axis=(0,1,2))
    (62.99, 62.09, 66.70),  # equals np.std(cifar10()['train']['data'], axis=(0,1,2))
]

CIFAR10_CLASSES = 'airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck'.split(', ')


#####################
# dict utils
#####################

def union(*dicts):
    return {k: v for d in dicts for (k, v) in d.items()}


def make_tuple(path):
    return (path,) if isinstance(path, str) else path


def map_values(func, dct):
    return {k: func(v) for k, v in dct.items()}


def path_iter(nested_dict, pfx=()):
    for name, val in nested_dict.items():
        if isinstance(val, dict):
            yield from path_iter(val, pfx + make_tuple(name))
        else:
            yield (pfx + make_tuple(name), val)


def map_nested(func, nested_dict):
    return {k: map_nested(func, v) if isinstance(v, dict) else func(v) for k, v in nested_dict.items()}


def group_by_key(seq):
    res = defaultdict(list)
    for k, v in seq:
        res[k].append(v)
    return res


def reorder(dct, keys):
    return {k: dct[k] for k in keys}


def mean_members(dictionary):
    """Returns a dictionary with the same keys and the mean of the values"""
    return {k: np.mean(v) for k, v in dictionary.items()}


def identity(value):
    return value


def map_types(mapping, net):
    def f(node):
        typ, *rest = node
        return (mapping.get(typ, typ), *rest)

    return map_nested(f, net)


def to(*args, **kwargs):
    """
    Returns a closure that applies x.to(*args, **kwargs) to x
    """
    def apply_to(x):
        return x.to(*args, **kwargs)
    return apply_to


#####################
# graph building
#####################


def build_graph(net, path_map='_'.join):
    net = {path: node if len(node) is 3 else (*node, None) for path, node in path_iter(net)}
    default_inputs = chain([('input',)], net.keys())
    resolve_path = lambda path, pfx: pfx + path if (pfx + path in net or not pfx) else resolve_path(net, path, pfx[:-1])
    return {path_map(path): (typ, value, (
        [path_map(default)] if inputs is None else [path_map(resolve_path(make_tuple(k), path[:-1])) for k in inputs]))
            for (path, (typ, value, inputs)), default in zip(net.items(), default_inputs)}


# node definitions
empty_signature = inspect.Signature()


class node_def(namedtuple('node_def', ['type'])):
    def __call__(self, *args, **kwargs):
        return (self.type, dict(signature(self.type).bind(*args, **kwargs).arguments))


#####################
# Layers
#####################

class Add(namedtuple('Add', [])):
    def __call__(self, x, y): return x + y


class AddWeighted(namedtuple('AddWeighted', ['wx', 'wy'])):
    def __call__(self, x, y): return self.wx * x + self.wy * y


class Identity(namedtuple('Identity', [])):
    def __call__(self, x): return x


class BatchNorm(nn.BatchNorm2d):
    def __init__(self, num_features, eps=1e-05, momentum=0.1, weight=True, bias=True):
        super().__init__(num_features, eps=eps, momentum=momentum)
        self.weight.data.fill_(1.0)
        self.bias.data.fill_(0.0)
        self.weight.requires_grad = weight
        self.bias.requires_grad = bias


class GhostBatchNorm(BatchNorm):
    def __init__(self, num_features, num_splits, **kw):
        super().__init__(num_features, **kw)
        self.num_splits = num_splits
        self.register_buffer('running_mean', torch.zeros(num_features * self.num_splits))
        self.register_buffer('running_var', torch.ones(num_features * self.num_splits))

    def train(self, mode=True):
        if (self.training is True) and (mode is False):  # lazily collate stats when we are going to use them
            self.running_mean = torch.mean(self.running_mean.view(self.num_splits, self.num_features), dim=0).repeat(
                self.num_splits)
            self.running_var = torch.mean(self.running_var.view(self.num_splits, self.num_features), dim=0).repeat(
                self.num_splits)
        return super().train(mode)

    def forward(self, input):
        N, C, H, W = input.shape
        if self.training or not self.track_running_stats:
            return F.batch_norm(
                input.view(-1, C * self.num_splits, H, W), self.running_mean, self.running_var,
                self.weight.repeat(self.num_splits), self.bias.repeat(self.num_splits),
                True, self.momentum, self.eps).view(N, C, H, W)
        else:
            return F.batch_norm(
                input, self.running_mean[:self.num_features], self.running_var[:self.num_features],
                self.weight, self.bias, False, self.momentum, self.eps)


class Mul(nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.weight = weight

    def __call__(self, x):
        return x * self.weight


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), x.size(1))


# Losses
class CrossEntropyLoss(namedtuple('CrossEntropyLoss', [])):
    def __call__(self, log_probs, target):
        return torch.nn.functional.nll_loss(log_probs, target, reduction='none')


class KLLoss(namedtuple('KLLoss', [])):
    def __call__(self, log_probs):
        return -log_probs.mean(dim=1)


class Correct(namedtuple('Correct', [])):
    def __call__(self, classifier, target):
        return classifier.max(dim=1)[1] == target


class LogSoftmax(namedtuple('LogSoftmax', ['dim'])):
    def __call__(self, x):
        return torch.nn.functional.log_softmax(x, self.dim, _stacklevel=5)


conv = node_def(nn.Conv2d)
linear = node_def(nn.Linear)
batch_norm = node_def(BatchNorm)
pool = node_def(nn.MaxPool2d)
relu = node_def(nn.ReLU)


def conv_block(c_in, c_out):
    return {
        'conv': conv(
            in_channels=c_in, out_channels=c_out, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
        ),
        'norm': batch_norm(c_out),
        'act': relu(),
    }


def conv_pool_block( c_in, c_out):
    return dict(conv_block(c_in, c_out), pool=pool(2))


def conv_pool_block_pre(c_in, c_out):
    return reorder(conv_pool_block(c_in, c_out), ('conv', 'pool', 'norm', 'act'))


def residual(c, conv_block):
    return {
        'in': (Identity, {}),
        'res1': conv_block(c, c),
        'res2': conv_block(c, c),
        'out': (Identity, {}),
        'add': (Add, {}, ['in', 'out']),
    }


def build_network(channels, extra_layers, res_layers, scale, conv_block=conv_block,
                  prep_block=conv_block, conv_pool_block=conv_pool_block, types=None):
    net = {
        'prep': prep_block(3, channels['prep']),
        'layer1': conv_pool_block(channels['prep'], channels['layer1']),
        'layer2': conv_pool_block(channels['layer1'], channels['layer2']),
        'layer3': conv_pool_block(channels['layer2'], channels['layer3']),
        'pool': pool(4),
        'classifier': {
            'flatten': (Flatten, {}),
            'conv': linear(channels['layer3'], 10, bias=False),
            'scale': (Mul, {'weight': scale}),
        },
        'logits': (Identity, {}),
    }
    for layer in res_layers:
        net[layer]['residual'] = residual(channels[layer], conv_block)
    for layer in extra_layers:
        net[layer]['extra'] = conv_block(channels[layer], channels[layer])
    if types: net = map_types(types, net)
    return net


def label_smoothing_loss(alpha):
    return Network({
        'logprobs': (LogSoftmax, {'dim': 1}, ['logits']),
        'KL': (KLLoss, {}, ['logprobs']),
        'xent': (CrossEntropyLoss, {}, ['logprobs', 'target']),
        'loss': (AddWeighted, {'wx': 1 - alpha, 'wy': alpha}, ['xent', 'KL']),
        'acc': (Correct, {}, ['logits', 'target']),
    })


def whitening_block(c_in, c_out, eigen_values=None, eigen_vectors=None, eps=1e-2):
    filt = nn.Conv2d(3, 27, kernel_size=(3, 3), padding=(1, 1), bias=False)
    filt.weight.data = (eigen_vectors / torch.sqrt(eigen_values + eps)[:, None, None, None])
    filt.weight.requires_grad = False

    return {
        'whiten': (identity, {'value': filt}),
        'conv': conv(27, c_out, kernel_size=(1, 1), bias=False),
        'norm': batch_norm(c_out),
        'act': relu(),
    }


######################
# Evaluation functions
######################

def forward_tta(tta_transforms, model, batch, loss):
    """
    Forward pass with test time augmentation
    """
    if model.training:
        model.train(False)
    logits = torch.mean(torch.stack(
        [model({'input': transform(batch['input'].clone())})['logits'].detach() for transform in tta_transforms],
        dim=0), dim=0)
    return loss(dict(batch, logits=logits))


def eval_on_batches(model, loss, vbatches):
    eval_log = []
    model.eval()
    for tb in vbatches:
        out = forward_tta([identity, flip_lr], model, tb, loss)
        eval_log.append(('loss', out['loss'].detach()))
        eval_log.append(('acc', out['acc'].detach()))
    # Average the activations
    res = map_values((lambda xs: to_numpy(torch.cat(xs)).astype(np.float)), group_by_key(eval_log))
    valid_summary = mean_members(res)
    return valid_summary


def save_log_to_tsv(log, path):
    with open(path, 'w') as f:
        tsv_writer = csv.writer(f, delimiter='\t')
        tsv_writer.writerow(['epochs', 'hours', 'top1Accuracy'])
        for epoch, l in enumerate(log):
            # Save the time in seconds and the accuracy as a percentage
            tsv_writer.writerow([epoch, l['time']/3600.0, l['valid']['acc']*100])


#####################
# Compat
#####################

class Network(nn.Module):
    def __init__(self, net, loss=None):
        super().__init__()
        self.graph = {path: (typ, typ(**params), inputs) for path, (typ, params, inputs) in build_graph(net).items()}
        self.loss = loss or identity
        for path, (_, node, _) in self.graph.items():
            setattr(self, path, node)

    def nodes(self):
        return (node for _, node, _ in self.graph.values())

    def forward(self, inputs):
        outputs = dict(inputs)
        for k, (_, node, ins) in self.graph.items():
            outputs[k] = node(*[outputs[x] for x in ins])
        return outputs

    def half(self):
        for node in self.nodes():
            if isinstance(node, nn.Module) and not isinstance(node, nn.BatchNorm2d):
                node.half()
        return self


def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return x


def flip_lr(x):
    if isinstance(x, torch.Tensor):
        return torch.flip(x, [-1])
    return x[..., ::-1].copy()


def trainable_params(model):
    return {k: p for k, p in model.named_parameters() if p.requires_grad}


#####################
# Optimisers
#####################


def nesterov_update(w, dw, v, lr, weight_decay, momentum):
    dw.add_(weight_decay, w).mul_(-lr)
    v.mul_(momentum).add_(dw)
    w.add_(dw.add_(momentum, v))


def zeros_like(weights):
    return [torch.zeros_like(w) for w in weights]


class SGDOpt(object):
    """
    A class to hold the optimizer state
    """
    def __init__(self,
                 weight_param_schedule,
                 bias_param_schedule,
                 weight_params,
                 bias_params):

        self.weights = weight_params
        self.bias = bias_params
        self._w_param_schedule = weight_param_schedule
        self._b_param_schedule = bias_param_schedule
        # Internal optimizer state initialization
        self.opt_state = zeros_like(self.weights)
        self.bias_opt_state = zeros_like(self.bias)
        self.step_number = 0
        self.update = nesterov_update
        self.last_step_b_parameters = {}
        self.last_step_w_parameters = {}

    def step(self):
        self.step_number += 1
        # The weights
        param_values = {k: f(self.step_number) for k, f in self._w_param_schedule.items()}
        self.last_step_w_parameters = param_values
        for w, v in zip(self.weights, self.opt_state):
            if w.requires_grad:
                self.update(w.data, w.grad.data, v, **param_values)

        param_values = {k: f(self.step_number) for k, f in self._b_param_schedule.items()}
        self.last_step_b_parameters = param_values
        for w, v in zip(self.bias, self.bias_opt_state):
            if w.requires_grad:
                self.update(w.data, w.grad.data, v, **param_values)


####################
# Hyperparameter Schedules
####################


class PiecewiseLinear(namedtuple('PiecewiseLinear', ('knots', 'vals'))):
    def __call__(self, t):
        return np.interp([t], self.knots, self.vals)[0]


class Const(namedtuple('Const', ['val'])):
    def __call__(self, x):
        return self.val


def lr_schedule(knots, vals, batch_size, batch_count):
    return PiecewiseLinear(np.array(knots) * batch_count, np.array(vals) / batch_size)


#####################
# DATA
#####################

@cache(None)
def cifar10(root='./data'):
    download = lambda train: torchvision.datasets.CIFAR10(root=root, train=train, download=True)
    return {k: {'data': torch.tensor(v.data), 'targets': torch.tensor(v.targets)}
            for k, v in [('train', download(True)), ('valid', download(False))]}


def normalise(data, mean, std):
    return (data - mean) / std


def pad(data, border):
    return nn.ReflectionPad2d(border)(data)


def transpose(x, source='NHWC', target='NCHW'):
    return x.permute([source.index(d) for d in target])


def preprocess(dataset, transforms):
    dataset = copy.copy(dataset)
    for transform in reversed(transforms):
        dataset['data'] = transform(dataset['data'])
    return dataset


def compute_patch_whitening_statistics(train_set):

    def cov(X):
        X = X/np.sqrt(X.size(0) - 1)
        return X.t() @ X

    def patches(data, patch_size=(3, 3), dtype=torch.float32):
        h, w = patch_size
        c = data.size(1)
        return data.unfold(2, h, 1).unfold(3, w, 1).transpose(1, 3).reshape(-1, c, h, w).to(dtype)

    def eigens(patches):
        n, c, h, w = patches.shape
        covariance = cov(patches.reshape(n, c*h*w))
        eigen_values, eigen_vectors = torch.symeig(covariance, eigenvectors=True)
        return eigen_values.flip(0), eigen_vectors.t().reshape(c*h*w, c, h, w).flip(0)

    eigen_values, eigen_vectors = eigens(
        patches(train_set['data'][:10000, :, 4:-4, 4:-4])
    ) # center crop to remove padding
    return eigen_values, eigen_vectors


def chunks(data, splits):
    return (data[start:end] for (start, end) in zip(splits, splits[1:]))


def even_splits(N, num_chunks):
    return np.cumsum(
        [0] + [(N // num_chunks) + 1] * (N % num_chunks) + [N // num_chunks] * (num_chunks - (N % num_chunks))
    )


def shuffled(xs, inplace=False):
    xs = xs if inplace else copy.copy(xs)
    np.random.shuffle(xs)
    return xs


def transformed(data, targets, transform, max_options=None, unshuffle=False, device=None):
    i = torch.randperm(len(data), device=device)
    data = data[i]
    options = shuffled(transform.options(data.shape), inplace=True)[:max_options]
    data = torch.cat([transform.apply(x, **choice) for choice, x in
                      zip(options, chunks(data, even_splits(len(data), len(options))))])
    return (data[torch.argsort(i)], targets) if unshuffle else (data, targets[i])


class Batches():
    """
    An iterable that returns batches of data
    """
    def __init__(self, batch_size, transforms=(), dataset=None, shuffle=True, drop_last=False, max_options=None,
                 device=None):
        self.dataset, self.transforms, self.shuffle, self.max_options = dataset, transforms, shuffle, max_options
        self.device = device
        # Shard data per worker
        N = len(dataset['data'])
        self.splits = list(range(0, N + 1, batch_size))
        if not drop_last and self.splits[-1] != N:
            self.splits.append(N)

    def __iter__(self):
        data, targets = self.dataset['data'], self.dataset['targets']
        for transform in self.transforms:
            data, targets = transformed(data, targets, transform, max_options=self.max_options,
                                        unshuffle=not self.shuffle, device=self.device)

        if self.shuffle:
            i = torch.randperm(len(data), device=self.device)
            data, targets = data[i], targets[i]

        return ({'input': x.clone(), 'target': y} for (x, y) in
                zip(chunks(data, self.splits), chunks(targets, self.splits)))

    def __len__(self):
        return len(self.splits) - 1


#####################
# Augmentations
#####################

class Crop(namedtuple('Crop', ('h', 'w'))):
    def apply(self, x, x0, y0):
        return x[..., y0:y0 + self.h, x0:x0 + self.w]

    def options(self, shape):
        *_, H, W = shape
        return [{'x0': x0, 'y0': y0} for x0 in range(W + 1 - self.w) for y0 in range(H + 1 - self.h)]


class FlipLR(namedtuple('FlipLR', ())):
    def apply(self, x, choice):
        return flip_lr(x) if choice else x

    def options(self, shape):
        return [{'choice': b} for b in [True, False]]


class Cutout(namedtuple('Cutout', ('h', 'w'))):
    def apply(self, x, x0, y0):
        x[..., y0:y0 + self.h, x0:x0 + self.w] = 0.0
        return x

    def options(self, shape):
        *_, H, W = shape
        return [{'x0': x0, 'y0': y0} for x0 in range(W + 1 - self.w) for y0 in range(H + 1 - self.h)]


#####################
# Timing
#####################


class Timer(object):
    def __init__(self, synch=None):
        self.synch = synch or (lambda: None)
        self.synch()
        self.times = [time.perf_counter()]
        self.total_time = 0.0

    def __call__(self, update_total=True):
        self.synch()
        self.times.append(time.perf_counter())
        delta_t = self.times[-1] - self.times[-2]
        if update_total:
            self.total_time += delta_t
        return delta_t
