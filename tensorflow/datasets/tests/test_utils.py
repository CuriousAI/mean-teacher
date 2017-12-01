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
