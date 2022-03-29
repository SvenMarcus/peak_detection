import numpy as np
from typing import Tuple, List
from numpy import ndarray
from scipy import interpolate
from scipy.stats import linregress
from consts import (LOCAL_POINTS, SLOPE_INITIAL_TILT, TILTING_STEPS, INTERSECTION_NUMBER)
from matplotlib import pyplot as plt

def median_distance(a: ndarray) -> ndarray:
    """
    Gets median distance between peaks
    :param a:
    :return: median
    """
    distances = []
    for i in range(len(a) - 1):
        distances.append(a[i] - a[i + 1])
    return np.median(distances)


def get_left_right_bound(y: ndarray, x: ndarray, px: float, left_crop: int) -> Tuple[int, int]:
    """
    Computes the left and right bounds of a peak, analysing the first derivative.
    First it interpolates the signal to be able to use it as a function.
    Then fixes a line through the point corresponding to the max of the peak in the
    original signal. The line is initialized close to a slope computed locally around the point.
    Finally it's rotated contra clock wise, as long as it has three intersection points with
    the interpolated functions.
    :param y: first derivative of the signal
    :param x: x values of the signal
    :param px: x of the maxima point
    :param left_crop: left cropping of the signal (for shifting)
    :return: left and right intersection
    """
    y_inter = interpolate.interp1d(x, y)

    px_i = px - left_crop
    py = y[px_i]

    slope = get_local_slope(px_i, x, y)
    m = np.linspace(slope + SLOPE_INITIAL_TILT, 0, TILTING_STEPS, endpoint=False)
    current_slope = slope

    for mi in m:
        c = py - (mi * px)
        line = lambda x_: mi * x_ + c
        intersections = get_inversion_idx(line(x) - y_inter(x))
        if len(intersections) == INTERSECTION_NUMBER:
            current_slope = mi
        else:
            break

    final_line = lambda x_: current_slope * x_ + c
    intersections = get_inversion_idx(final_line(x) - y_inter(x))
    plt.plot(final_line(x))
    plt.plot(y_inter(x))
    plt.show()
    inter = (intersections[0], intersections[-1])
    return inter


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


def get_local_slope(px_i: float, x: ndarray, y: ndarray) -> float:
    """
    Computes the local slope around a point
    :param px_i: x value of the point
    :param x: array of x values
    :param y: array of y values
    :return: the slope
    """

    p_neigh_x = x[px_i - LOCAL_POINTS:px_i + LOCAL_POINTS]
    p_neigh_y = y[px_i - LOCAL_POINTS:px_i + LOCAL_POINTS]
    slope, intercept, r, p, se = linregress(p_neigh_x, p_neigh_y)
    return slope
