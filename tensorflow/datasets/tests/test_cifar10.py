# Copyright (c) 2018, Curious AI Ltd. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

import numpy as np

from ..cifar10 import Cifar10ZCA


def test_supervised_testing():
    cifar = Cifar10ZCA(test_phase=True)
    assert_shapes(cifar.training, 50000)
    assert_label_range(cifar.training, range(10))
    assert_shapes(cifar.evaluation, 10000)
    assert_label_range(cifar.evaluation, range(10))


def test_semisupervised_testing():
    cifar = Cifar10ZCA(test_phase=True, n_labeled=100)
    assert_shapes(cifar.training, 50000)
    assert_label_range(cifar.training, range(-1, 10))
    assert_label_distribution(cifar.training, [49900] + [10] * 10)
    assert_shapes(cifar.evaluation, 10000)
    assert_label_range(cifar.evaluation, range(10))


def test_supervised_validation():
    cifar = Cifar10ZCA()
    assert_shapes(cifar.training, 45000)
    assert_label_range(cifar.training, range(10))
    assert_shapes(cifar.evaluation, 5000)
    assert_label_range(cifar.evaluation, range(10))


def test_semisupervised_validation():
    cifar = Cifar10ZCA(n_labeled=100)
    assert_shapes(cifar.training, 45000)
    assert_label_range(cifar.training, range(-1, 10))
    assert_label_distribution(cifar.training, [44900] + [10] * 10)
    assert_shapes(cifar.evaluation, 5000)
    assert_label_range(cifar.evaluation, range(10))


def assert_shapes(data, n_expected_examples):
    assert data['x'].shape == (n_expected_examples, 32, 32, 3)
    assert data['y'].shape == (n_expected_examples,)


def assert_label_range(data, expected_range):
    assert np.min(data['y']) == min(expected_range)
    assert np.max(data['y']) == max(expected_range)


def assert_label_distribution(data, expected_distribution):
    label_distribution = np.bincount(data['y'] + 1)
    assert label_distribution.tolist() == expected_distribution
