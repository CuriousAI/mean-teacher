# Copyright (c) 2018, Curious AI Ltd. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

import numpy as np

from ..utils import random_balanced_partitions


def test_random_balanced_partition():
    results = [
        random_balanced_partitions(np.array(['a', 'b', 'c']), 2, [3, 5, 5])
        for _ in range(100)
    ]
    results = [(a.tolist(), b.tolist()) for (a, b) in results]
    assert (['a', 'b'], ['c']) in results
    assert (['a', 'c'], ['b']) in results
    assert not (['b', 'c'], ['a']) in results
