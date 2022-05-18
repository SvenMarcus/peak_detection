from dataclasses import dataclass
from typing import List, Tuple, TypeVar, Union

import numpy as np
import numpy.typing as npt
from scipy import interpolate
from scipy.signal import find_peaks

from consts import INTERSECTION_NUMBER, LOCAL_POINTS, SLOPE_INITIAL_TILT, TILTING_STEPS
from utils import get_inversion_idx, get_local_slope, median_distance

ValueArray = npt.NDArray[np.float64]
PositionArray = npt.NDArray[np.int64]
GenericArrayType = TypeVar("GenericArrayType", ValueArray, PositionArray)

PeakBound = Tuple[int, int]


@dataclass
class BoundedPeak:
    bounds: PeakBound
    position: int


def peak_series(values: ValueArray) -> List[BoundedPeak]:
    gradients = np.gradient(values, 1)
    peak_positions, _ = find_peaks(values, prominence=1)
    estimated_bounds = _estimate_bounds(values, peak_positions)
    accurate_bounds = _collect_accurate_bounds(
        gradients, peak_positions, estimated_bounds
    )
    zipped = zip(accurate_bounds, peak_positions)
    return [BoundedPeak(bounds, p) for bounds, p in zipped]


def _estimate_bounds(
    values: ValueArray, peak_positions: PositionArray
) -> List[PeakBound]:
    avg_peak_distance = _average_peak_distances(peak_positions)
    left_bound_estimates = np.clip(peak_positions - avg_peak_distance, 0, len(values))
    righ_bound_estimates = np.clip(peak_positions + avg_peak_distance, 0, len(values))
    return list(zip(left_bound_estimates, righ_bound_estimates))


def _average_peak_distances(peak_positions: PositionArray) -> int:
    return abs(int(median_distance(peak_positions) / 2))


def _collect_accurate_bounds(
    gradients: ValueArray,
    peak_positions: PositionArray,
    estimated_bounds: List[PeakBound],
) -> List[PeakBound]:
    return [
        _accurate_bounds_for_peak(gradients, estimated_bounds[i], p)
        for i, p in enumerate(peak_positions)
    ]


def _accurate_bounds_for_peak(
    gradients: ValueArray, peak_bound_estimate: PeakBound, peak_position: int
) -> PeakBound:
    return _get_left_right_bound(
        gradients=gradients[peak_bound_estimate[0] : peak_bound_estimate[1]],
        peak_position=peak_position,
        left_crop=peak_bound_estimate[0],
    )


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
    :param y: first derivative of the signal
    :param px: x of the maxima point
    :param left_crop: left cropping of the signal (for shifting)
    :return: left and right intersection
    """
    x: PositionArray = np.arange(0, len(gradients), 1, dtype=np.int64)
    cropped_peak_position = peak_position - left_crop
    intersections = _find_relevant_intersections(x, gradients, cropped_peak_position)
    print(len(intersections))
    return (intersections[0] + left_crop, intersections[-1] + left_crop)


def _get_slope_tilts(
    x: PositionArray, gradients: ValueArray, peak_position: float
) -> ValueArray:
    slope = get_local_slope(peak_position, x, gradients, LOCAL_POINTS)
    return np.linspace(slope + SLOPE_INITIAL_TILT, 0, TILTING_STEPS, endpoint=False)


def _find_relevant_intersections(
    x: PositionArray, gradients: ValueArray, peak_position: float
) -> List[int]:
    slope_tilts = _get_slope_tilts(x, gradients, peak_position)
    peak_gradient = gradients[peak_position]
    print(peak_gradient)
    for current_slope in slope_tilts:
        y_intercept = _get_y_intercept(peak_position, peak_gradient, current_slope)
        function = _linear_function(x, current_slope, y_intercept)
        intersections = _intersections_of_functions(gradients, function)

        if len(intersections) != INTERSECTION_NUMBER:
            break
    return intersections


def _get_y_intercept(x: float, y: float, slope: float) -> float:
    return y - (slope * x)


def _intersections_of_functions(
    gradients: GenericArrayType, current_function: GenericArrayType
) -> List[int]:
    return get_inversion_idx(current_function - gradients)


def _linear_function(x: GenericArrayType, slope: float, y_intercept: float) -> ValueArray:
    return x * slope + y_intercept  # type: ignore
