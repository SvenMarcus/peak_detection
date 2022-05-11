from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import numpy.typing as npt
from scipy.signal import find_peaks

from utils import get_left_right_bound, median_distance


ValueArray = npt.NDArray[np.float64]
PositionArray = npt.NDArray[np.int64]

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
    bounds = get_left_right_bound(
        y=gradients[peak_bound_estimate[0] : peak_bound_estimate[1]],
        px=peak_position,
        left_crop=peak_bound_estimate[0],
    )

    return bounds
