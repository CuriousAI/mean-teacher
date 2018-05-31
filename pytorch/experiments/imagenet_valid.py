# Copyright (c) 2018, Curious AI Ltd. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

"""Train ImageNet with 10% of the labels and evaluate against the validation set"""

import sys
import logging

import torch

import main
from mean_teacher.cli import parse_dict_args
from mean_teacher.run_context import RunContext


LOG = logging.getLogger('runner')


def parameters():
    defaults = {
        # Technical details
        'workers': 20,
        'checkpoint_epochs': 1,
        'evaluation_epochs': 1,

        # Data
        'dataset': 'imagenet',
        'exclude_unlabeled': False,

        # Data sampling
        'base_batch_size': 40,
        'base_labeled_batch_size': 20,

        # Architecture
        'arch': 'resnext152',
        'ema_decay': .9997,

        # Costs
        'consistency_type': 'kl',
        'consistency': 10.0,
        'consistency_rampup': 5,
        'logit_distance_cost': 0.01,
        'weight_decay': 5e-5,

        # Optimization
        'epochs': 60,
        'lr_rampdown_epochs': 75,
        'lr_rampup': 2,
        'initial_lr': 0.1,
        'base_lr': 0.025,
        'nesterov': True,
    }

    for data_seed in range(10, 12):
        yield {
            **defaults,
            'title': 'mean teacher r-152 eval',
            'data_seed': 0
        }


def run(title, base_batch_size, base_labeled_batch_size, base_lr, data_seed, **kwargs):
    LOG.info('run title: %s', title)
    ngpu = torch.cuda.device_count()
    adapted_args = {
        'batch_size': base_batch_size * ngpu,
        'labeled_batch_size': base_labeled_batch_size * ngpu,
        'lr': base_lr * ngpu,
        'labels': 'data-local/labels/ilsvrc2012/128000_balanced_labels/{:02d}.txt'.format(data_seed),
    }
    context = RunContext(__file__, data_seed)
    main.args = parse_dict_args(**adapted_args, **kwargs)
    main.main(context)


if __name__ == "__main__":
    for run_params in parameters():
        run(**run_params)
