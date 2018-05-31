# Copyright (c) 2018, Curious AI Ltd. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

"""SVHN supervised evaluation"""

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
    for n_labeled in [250, 500, 1000]:
        for data_seed in range(10):
            yield {
                'test_phase': test_phase,
                'n_labeled': n_labeled,
                'data_seed': data_seed,
                'training_length': 40000,
                'rampdown_length': 10000
            }

    for data_seed in range(4):
        yield {
            'test_phase': test_phase,
            'n_labeled': 'all',
            'data_seed': data_seed,
            'training_length': 180000,
            'rampdown_length': 25000
        }



def run(test_phase, data_seed, n_labeled, training_length, rampdown_length):
    minibatch_size = 100
    n_labeled_per_batch = 100

    tf.reset_default_graph()
    model = Model(RunContext(__file__, data_seed))

    cifar = SVHN(n_labeled=n_labeled,
                 data_seed=data_seed,
                 test_phase=test_phase)

    model['ema_consistency'] = True
    model['max_consistency_cost'] = 0.0
    model['apply_consistency_to_labeled'] = False
    model['rampdown_length'] = rampdown_length
    model['training_length'] = training_length

    # Turn off augmentation
    model['translate'] = False
    model['flip_horizontally'] = False

    training_batches = minibatching.training_batches(cifar.training,
                                                     minibatch_size,
                                                     n_labeled_per_batch)
    evaluation_batches_fn = minibatching.evaluation_epoch_generator(cifar.evaluation,
                                                                    minibatch_size)

    tensorboard_dir = model.save_tensorboard_graph()
    LOG.info("Saved tensorboard graph to %r", tensorboard_dir)

    model.train(training_batches, evaluation_batches_fn)


if __name__ == "__main__":
    for run_params in parameters():
        run(**run_params)
