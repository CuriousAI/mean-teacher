# Copyright (c) 2018, Curious AI Ltd. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

"""Vary ema decay parameter on 250-label SVHN for the NIPS paper"""

import logging
import sys

from .run_context import RunContext
import tensorflow as tf

from datasets import SVHN
from mean_teacher.model import Model
from mean_teacher import minibatching


LOG = logging.getLogger('main')


def parameters():
    n_runs = 4
    for data_seed in range(1000, 1000 + n_runs):
        for max_consistency_cost in [0, 0.1, 0.3, 1, 3, 10, 30]:
            yield {
                'data_seed': data_seed,
                'max_consistency_cost': max_consistency_cost
            }


def model_hyperparameters(model_type, n_labeled, n_extra_unlabeled):
    assert model_type in ['mean_teacher', 'pi']
    training_length = {
        0: 180000,
        100000: 400000,
        500000: 600000,
    }
    if n_labeled == 'all':
        return {
            'training_length': training_length[n_extra_unlabeled],
            'n_labeled_per_batch': 100,
            'max_consistency_cost': 100.0,
            'apply_consistency_to_labeled': True,
            'ema_consistency': model_type == 'mean_teacher'
        }
    elif isinstance(n_labeled, int):
        return {
            'training_length': training_length[n_extra_unlabeled],
            'n_labeled_per_batch': 1,
            'max_consistency_cost': 1.0,
            'apply_consistency_to_labeled': False,
            'ema_consistency': model_type == 'mean_teacher'
        }
    else:
        msg = "Unexpected combination: {model_type}, {n_labeled}, {n_extra_unlabeled}"
        assert False, msg.format(locals())


def run(data_seed, max_consistency_cost,
        test_phase=False, n_labeled=250, n_extra_unlabeled=0, model_type='mean_teacher'):
    minibatch_size = 100
    hyperparams = model_hyperparameters(model_type, n_labeled, n_extra_unlabeled)

    tf.reset_default_graph()
    model = Model(RunContext(__file__, data_seed))

    svhn = SVHN(n_labeled=n_labeled,
                n_extra_unlabeled=n_extra_unlabeled,
                data_seed=data_seed,
                test_phase=test_phase)

    model['ema_consistency'] = hyperparams['ema_consistency']
    model['apply_consistency_to_labeled'] = hyperparams['apply_consistency_to_labeled']
    model['training_length'] = hyperparams['training_length']
    #model['max_consistency_cost'] = hyperparams['max_consistency_cost']
    model['max_consistency_cost'] = max_consistency_cost

    training_batches = minibatching.training_batches(svhn.training,
                                                     minibatch_size,
                                                     hyperparams['n_labeled_per_batch'])
    evaluation_batches_fn = minibatching.evaluation_epoch_generator(svhn.evaluation,
                                                                    minibatch_size)

    tensorboard_dir = model.save_tensorboard_graph()
    LOG.info("Saved tensorboard graph to %r", tensorboard_dir)

    model.train(training_batches, evaluation_batches_fn)


if __name__ == "__main__":
    for run_params in parameters():
        run(**run_params)
