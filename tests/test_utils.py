import numpy as np
from utils import (median_distance, get_inversion_idx, get_local_slope)


def test_median_distance():
    array = np.array([0, 1, 2, 3, 4, 18])
    true_median = 1.
    test_median = median_distance(array)
    assert test_median == true_median


def test_get_inversion_idx():
    array = np.array([-1, -1, -1, 1, 1, 1])
    true_idx = 3
    test_idx = get_inversion_idx(array)[0]
    assert test_idx == true_idx


def test_get_local_slope():
    x = np.array([0, 1, 2, 3, 4])
    y = np.array([0, 1, 2, 3, 4])
    p = 2
    neighbor_points = 2
    true_slope = 1.
    assert get_local_slope(p, x, y, neighbor_points) == true_slope
