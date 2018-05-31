# Copyright (c) 2018, Curious AI Ltd. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

from itertools import islice

import numpy as np

from ..minibatching import (combine_batches, eternal_batches,
                            eternal_random_index_batches,
                            training_batches)


def test_indexes():
    batch_generator = eternal_random_index_batches(max_index=20, batch_size=5)

    first_epoch = list(list(batch) for batch in islice(batch_generator, 4))
    second_epoch = list(list(batch) for batch in islice(batch_generator, 4))
    first_epoch_combined = [item for batch in first_epoch for item in batch]
    second_epoch_combined = [item for batch in second_epoch for item in batch]

    assert (len(batch) == 5 for batch in first_epoch)
    assert (len(batch) == 5 for batch in second_epoch)
    assert sorted(first_epoch_combined) == list(range(20))
    assert sorted(second_epoch_combined) == list(range(20))
    assert first_epoch_combined != second_epoch_combined


def test_indexes_uneven_epoch():
    batch_generator = eternal_random_index_batches(max_index=5, batch_size=2)

    first_two_epochs = list(list(batch) for batch in islice(batch_generator, 5))
    first_two_epochs_combined = [item for batch in first_two_epochs for item in batch]
    first_half = first_two_epochs_combined[:5]
    second_half = first_two_epochs_combined[5:]

    assert sorted(first_half) == list(range(5))
    assert sorted(second_half) == list(range(5))


def test_eternal_training():
    data = np.array(['a', 'b', 'c'])

    batch_generator = eternal_batches(data, batch_size=2)

    first_two_epochs = list(islice(batch_generator, 3))
    assert [len(batch) for batch in first_two_epochs] == [2, 2, 2]

    first_two_epochs_combined = [item for batch in first_two_epochs for item in batch]
    first_half = first_two_epochs_combined[:3]
    second_half = first_two_epochs_combined[3:]

    assert sorted(first_half) == ['a', 'b', 'c']
    assert sorted(second_half) == ['a', 'b', 'c']


def test_batches_from_two_sets():
    data1 = np.array(['a', 'b'])
    data2 = np.array(['c', 'd', 'e'])

    batch_generator = combine_batches(
        eternal_batches(data1, batch_size=1),
        eternal_batches(data2, batch_size=2)
    )

    first_six_batches = list(islice(batch_generator, 6))
    assert [len(batch) for batch in first_six_batches] == [3, 3, 3, 3, 3, 3]

    batch_portions1 = [batch[:1] for batch in first_six_batches]
    batch_portions2 = [batch[1:] for batch in first_six_batches]

    returned1 = np.concatenate(batch_portions1)
    returned2 = np.concatenate(batch_portions2)

    epochs1 = np.split(returned1, 3)
    epochs2 = np.split(returned2, 4)

    assert all(sorted(items) == ['a', 'b'] for items in epochs1)
    assert all(sorted(items) == ['c', 'd', 'e'] for items in epochs2)


def test_stratified_batches():
    data = np.array([('a', -1), ('b', 0), ('c', 1), ('d', -1), ('e', -1)],
                    dtype=[('x', np.str_, 8), ('y', np.int32)])

    assert list(data['x']) == ['a', 'b', 'c', 'd', 'e']
    assert list(data['y']) == [-1, 0, 1, -1, -1]

    batch_generator = training_batches(data, batch_size=3, n_labeled_per_batch=1)

    first_ten_batches = list(islice(batch_generator, 10))

    labeled_batch_portions = [batch[:1] for batch in first_ten_batches]
    unlabeled_batch_portions = [batch[1:] for batch in first_ten_batches]

    labeled_epochs = np.split(np.concatenate(labeled_batch_portions), 5)
    unlabeled_epochs = np.split(np.concatenate(unlabeled_batch_portions), 4)

    assert ([sorted(items['x'].tolist()) for items in labeled_epochs] ==
            [['b', 'c']] * 5)
    assert ([sorted(items['y'].tolist()) for items in labeled_epochs] ==
            [[0, 1]] * 5)
    assert ([sorted(items['x'].tolist()) for items in unlabeled_epochs] ==
            [['a', 'b', 'c', 'd', 'e']] * 4)
    assert ([sorted(items['y'].tolist()) for items in unlabeled_epochs] ==
            [[-1, -1, -1, -1, -1]] * 4)
