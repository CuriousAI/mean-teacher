# Copyright (c) 2018, Curious AI Ltd. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

"""
    Creates static contrast normalized and ZCA-whitened dataset using the following parameters
    - global contrast normalization using Goodfellow scale factor 55.
    - ZCA using filter_bias=0.1
"""

import os
import numpy as np
from scipy.io import loadmat


DIR = os.path.join('data', 'images', 'cifar', 'cifar10')


def assert_not_exists(path):
    assert not os.path.exists(path), ""


def cifar10_orig_train():
    return load_batch_files([os.path.join(DIR, "data_batch_{}.mat".format(i)) for i in range(1, 6)])


def cifar10_orig_test():
    return load_batch_file(os.path.join(DIR, "test_batch.mat"))


def load_batch_files(batch_files):
    data_batches, label_batches = zip(*[load_batch_file(batch_file) for batch_file in batch_files])
    x = np.concatenate(data_batches, axis=0)
    y = np.concatenate(label_batches, axis=0)
    return x, y


def load_batch_file(path):
    d = loadmat(path)
    x = d['data'].astype(np.uint8)
    y = d['labels'].astype(np.uint8).flatten()
    return x, y


def to_channel_rgb(x):
    return np.transpose(np.reshape(x, (x.shape[0], 3, 32, 32)), [0, 2, 3, 1])


def global_contrast_normalize(X, scale=55., min_divisor=1e-8):
    X = X - X.mean(axis=1)[:, np.newaxis]

    normalizers = np.sqrt((X ** 2).sum(axis=1)) / scale
    normalizers[normalizers < min_divisor] = 1.

    X /= normalizers[:, np.newaxis]

    return X


def create_zca(imgs, filter_bias=0.1):
    meanX = np.mean(imgs, axis=0)

    covX = np.cov(imgs.T)
    D, E = np.linalg.eigh(covX + filter_bias * np.eye(covX.shape[0], covX.shape[1]))

    assert not np.isnan(D).any()
    assert not np.isnan(E).any()
    assert D.min() > 0

    D **= -.5

    W = np.dot(E, np.dot(np.diag(D), E.T))

    def transform(images):
        return np.dot(images - meanX, W)

    return transform


def do():
    train_x_orig, train_y = cifar10_orig_train()
    test_x_orig, test_y = cifar10_orig_test()

    train_x_gcn = global_contrast_normalize(train_x_orig)
    zca = create_zca(train_x_gcn)
    train_x = to_channel_rgb(zca(train_x_gcn))
    test_x = to_channel_rgb(zca(global_contrast_normalize(test_x_orig)))
    p = os.path.join(DIR, "cifar10_gcn_zca_v2.npz")
    assert_not_exists(p)
    np.savez(p, train_x=train_x, train_y=train_y, test_x=test_x, test_y=test_y)


if __name__ == "__main__":
    do()
