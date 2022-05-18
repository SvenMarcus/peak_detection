import numpy as np
from typing import Tuple, List
from numpy import ndarray
from scipy import interpolate
from scipy.stats import linregress


def median_distance(a: ndarray) -> ndarray:
    """
    Gets median distance between points
    :param a:
    :return: median
    """
    distances = []
    for i in range(len(a) - 1):
        distances.append(a[i] - a[i + 1])
    return abs(np.median(distances))


def get_inversion_idx(array: ndarray) -> List[int]:
    """
    Finds the points where the function changes sign
    :param array:
    :return:
    """
    a = np.sign(array)
    current_sign = a[0]
    idxs = []
    for i in range(len(a)):
        if current_sign * a[i] > 0:
            current_sign = a[i]
        elif current_sign * a[i] < 0:
            idxs.append(i)
            current_sign = a[i]
    return idxs


def get_local_slope(px_i: float, x: ndarray, y: ndarray, neighbor_points: int) -> float:
    """
    Computes the local slope around a point
    :param px_i: x value of the point
    :param x: array of x values
    :param y: array of y values
    :param neighbor_points: how many points before and after px_i to consider to compute the slope
    :return: the slope
    """

    p_neigh_x = x[px_i - neighbor_points : px_i + neighbor_points]
    p_neigh_y = y[px_i - neighbor_points : px_i + neighbor_points]
    slope, intercept, r, p, se = linregress(p_neigh_x, p_neigh_y)
    return slope
