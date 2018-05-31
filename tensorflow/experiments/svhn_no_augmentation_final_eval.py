# Copyright (c) 2018, Curious AI Ltd. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

"""SVHN final evaluation without augmentation"""

import logging
import sys

from .run_context import RunContext
import tensorflow as tf

from datasets import SVHN
from mean_teacher.model import Model
from mean_teacher import minibatching


LOG = logging.getLogger('main')


def parameters():
    test_phase = True
    for n_labeled in [500, 250, 1000, 'all', 100]:
        for n_extra_unlabeled in [0]:
            for model_type in ['mean_teacher', 'pi']:
                if n_extra_unlabeled > 0 and n_labeled != 500:
                    continue
                if n_labeled == 'all':
                    n_runs = 4
                else:
                    n_runs = 10
                for data_seed in range(2000, 2000 + n_runs):
                    yield {
                        'test_phase': test_phase,
                        'model_type': model_type,
                        'n_labeled': n_labeled,
                        'n_extra_unlabeled': n_extra_unlabeled,
                        'data_seed': data_seed
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


def run(test_phase, n_labeled, n_extra_unlabeled, data_seed, model_type):
    minibatch_size = 100
    hyperparams = model_hyperparameters(model_type, n_labeled, n_extra_unlabeled)

    tf.reset_default_graph()
    model = Model(RunContext(__file__, data_seed))

    svhn = SVHN(n_labeled=n_labeled,
                n_extra_unlabeled=n_extra_unlabeled,
                data_seed=data_seed,
                test_phase=test_phase)

    model['ema_consistency'] = hyperparams['ema_consistency']
    model['max_consistency_cost'] = hyperparams['max_consistency_cost']
    model['apply_consistency_to_labeled'] = hyperparams['apply_consistency_to_labeled']
    model['training_length'] = hyperparams['training_length']

    # Turn off augmentation
    model['translate'] = False
    model['flip_horizontally'] = False

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
