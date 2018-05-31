# Copyright (c) 2018, Curious AI Ltd. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

from itertools import islice, chain

import numpy as np


def evaluation_epoch_generator(data, batch_size=100):
    def generate():
        for idx in range(0, len(data), batch_size):
            yield data[idx:(idx + batch_size)]
    return generate


def training_batches(data, batch_size=100, n_labeled_per_batch='vary', random=np.random):
    if n_labeled_per_batch == 'vary':
        return eternal_batches(data, batch_size, random)
    elif n_labeled_per_batch == batch_size:
        labeled_data, _ = split_labeled(data)
        return eternal_batches(labeled_data, batch_size, random)
    else:
        assert 0 < n_labeled_per_batch < batch_size
        n_unlabeled_per_batch = batch_size - n_labeled_per_batch
        labeled_data, _ = split_labeled(data)
        return combine_batches(
            eternal_batches(labeled_data, n_labeled_per_batch, random),
            unlabel_batches(eternal_batches(data, n_unlabeled_per_batch, random))
        )


def split_labeled(data):
    is_labeled = (data['y'] != -1)
    return data[is_labeled], data[~is_labeled]


def combine_batches(*batch_generators):
    return (np.concatenate(batches) for batches in zip(*batch_generators))


def eternal_batches(data, batch_size=100, random=np.random):
    assert batch_size > 0 and len(data) > 0
    for batch_idxs in eternal_random_index_batches(len(data), batch_size, random):
        yield data[batch_idxs]


def unlabel_batches(batch_generator):
    for batch in batch_generator:
        batch["y"] = -1
        yield batch


def eternal_random_index_batches(max_index, batch_size, random=np.random):
    def random_ranges():
        while True:
            indices = np.arange(max_index)
            random.shuffle(indices)
            yield indices

    def batch_slices(iterable):
        while True:
            yield np.array(list(islice(iterable, batch_size)))

    eternal_random_indices = chain.from_iterable(random_ranges())
    return batch_slices(eternal_random_indices)
