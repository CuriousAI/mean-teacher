# Copyright (c) 2018, Curious AI Ltd. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

"""CIFAR-10 final evaluation without augmentation"""

import logging
import sys

from .run_context import RunContext
import tensorflow as tf

from datasets import Cifar10ZCA
from mean_teacher.model import Model
from mean_teacher import minibatching


LOG = logging.getLogger('main')


def parameters():
    test_phase = True
    for n_labeled in [4000, 'all', 1000, 2000]:
        for model_type in ['mean_teacher', 'pi']:
            if n_labeled == 'all':
                n_runs = 4
            else:
                n_runs = 10
            for data_seed in range(2000, 2000 + n_runs):
                yield {
                    'test_phase': test_phase,
                    'model_type': model_type,
                    'n_labeled': n_labeled,
                    'data_seed': data_seed
                }


def model_hyperparameters(model_type, n_labeled):
    assert model_type in ['mean_teacher', 'pi']
    if n_labeled == 'all':
        return {
            'n_labeled_per_batch': 100,
            'max_consistency_cost': 100.0,
            'apply_consistency_to_labeled': True,
            'ema_consistency': model_type == 'mean_teacher'
        }
    elif isinstance(n_labeled, int):
        return {
            'n_labeled_per_batch': 'vary',
            'max_consistency_cost': 100.0 * n_labeled / 50000,
            'apply_consistency_to_labeled': True,
            'ema_consistency': model_type == 'mean_teacher'
        }
    else:
        msg = "Unexpected combination: {model_type}, {n_labeled}"
        assert False, msg.format(locals())


def run(test_phase, n_labeled, data_seed, model_type):
    minibatch_size = 100
    hyperparams = model_hyperparameters(model_type, n_labeled)

    tf.reset_default_graph()
    model = Model(RunContext(__file__, data_seed))

    cifar = Cifar10ZCA(n_labeled=n_labeled,
                       data_seed=data_seed,
                       test_phase=test_phase)

    model['ema_consistency'] = hyperparams['ema_consistency']
    model['max_consistency_cost'] = hyperparams['max_consistency_cost']
    model['apply_consistency_to_labeled'] = hyperparams['apply_consistency_to_labeled']
    model['adam_beta_2_during_rampup'] = 0.999
    model['ema_decay_during_rampup'] = 0.999
    model['normalize_input'] = False  # Keep ZCA information
    model['rampdown_length'] = 25000
    model['training_length'] = 150000

    # Turn off augmentation
    model['translate'] = False
    model['flip_horizontally'] = False

    training_batches = minibatching.training_batches(cifar.training,
                                                     minibatch_size,
                                                     hyperparams['n_labeled_per_batch'])
    evaluation_batches_fn = minibatching.evaluation_epoch_generator(cifar.evaluation,
                                                                    minibatch_size)

    tensorboard_dir = model.save_tensorboard_graph()
    LOG.info("Saved tensorboard graph to %r", tensorboard_dir)

    model.train(training_batches, evaluation_batches_fn)


if __name__ == "__main__":
    for run_params in parameters():
        run(**run_params)
