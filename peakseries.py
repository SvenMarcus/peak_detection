from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import numpy.typing as npt
from scipy import interpolate
from scipy.signal import find_peaks

from consts import INTERSECTION_NUMBER, LOCAL_POINTS, SLOPE_INITIAL_TILT, TILTING_STEPS
from utils import get_inversion_idx, get_local_slope, median_distance

ValueArray = npt.NDArray[np.float64]
PositionArray = npt.NDArray[np.int64]

PeakBound = Tuple[int, int]


@dataclass
class BoundedPeak:
    bounds: PeakBound
    position: int


def peak_series(values: ValueArray) -> List[BoundedPeak]:
    peak_positions, _ = find_peaks(values, prominence=1)
    estimated = _find_estimated_bounded_peaks(values, peak_positions)
    return _find_accurate_bounded_peaks(values, estimated)


def _find_estimated_bounded_peaks(
    values: ValueArray, peak_positions: PositionArray
) -> List[BoundedPeak]:
    avg_peak_distance = _average_peak_distances(peak_positions)
    left_bound_estimates = np.clip(peak_positions - avg_peak_distance, 0, len(values))
    righ_bound_estimates = np.clip(peak_positions + avg_peak_distance, 0, len(values))

    zipped = zip(left_bound_estimates, righ_bound_estimates)
    return [
        BoundedPeak(estimated_bounds, peak_positions[p])
        for p, estimated_bounds in enumerate(zipped)
    ]


def _average_peak_distances(peak_positions: PositionArray) -> int:
    return abs(int(median_distance(peak_positions) / 2))


def _find_accurate_bounded_peaks(
    values: ValueArray, peak_estimates: List[BoundedPeak]
) -> List[BoundedPeak]:
    gradients = np.gradient(values, 1)
    for peak in peak_estimates:
        gradients_in_estimated_bounds = gradients[peak.bounds[0] : peak.bounds[1]]
        peak.bounds = _get_left_right_bound(
            gradients_in_estimated_bounds, peak.position, peak.bounds[0]
        )

    return peak_estimates


def _get_left_right_bound(
    gradients: ValueArray, peak_position: float, left_crop: int
) -> PeakBound:
    """
    Computes the left and right bounds of a peak, analysing the first derivative.
    First it interpolates the signal to be able to use it as a function.
    Then fixes a line through the point corresponding to the max of the peak in the
    original signal. The line is initialized close to a slope computed locally around the point.
    Finally it's rotated contra clock wise, as long as it has three intersection points with
    the interpolated functions.
    :param gradients: first derivative of the signal
    :param peak_position: x of the maxima point
    :param left_crop: left cropping of the signal (for shifting)
    :return: left and right intersection
    """
    x: PositionArray = np.arange(0, len(gradients), 1, dtype=np.int64)
    interpolated_gradients = _interpolate(x, gradients)
    cropped_peak_position = peak_position - left_crop
    intersections = _line_intersections_with_gradients(
        x, interpolated_gradients, cropped_peak_position
    )

    return intersections[0] + left_crop, intersections[-1] + left_crop


def _interpolate(x: PositionArray, y: ValueArray) -> ValueArray:
    interpolation = interpolate.interp1d(x, y)
    interpolated: ValueArray = interpolation(x)
    return interpolated


def _line_intersections_with_gradients(
    x: PositionArray, gradients: ValueArray, peak_position: float
) -> List[int]:
    slope_tilts = _get_slope_tilts(x, gradients, peak_position)
    peak_gradient = gradients[peak_position]

    final_intersections = [0, 0]
    for current_slope in slope_tilts:
        y_intercept = _get_y_intercept(peak_position, peak_gradient, current_slope)
        line = _linear_function(x, current_slope, y_intercept)
        intersections = _intersections_of_functions(gradients, line)

        if len(intersections) != INTERSECTION_NUMBER:
            break

        final_intersections = intersections

    return final_intersections


def _get_slope_tilts(
    x: PositionArray, gradients: ValueArray, peak_position: float
) -> ValueArray:
    slope = get_local_slope(peak_position, x, gradients, LOCAL_POINTS)
    return np.linspace(slope + SLOPE_INITIAL_TILT, 0, TILTING_STEPS, endpoint=False)


def _get_y_intercept(x: float, y: float, slope: float) -> float:
    return y - (slope * x)


def _linear_function(x: PositionArray, slope: float, y_intercept: float) -> ValueArray:
    return x * slope + y_intercept  # type: ignore


def _intersections_of_functions(
    gradients: ValueArray, current_function: ValueArray
) -> List[int]:
    return get_inversion_idx(current_function - gradients)
