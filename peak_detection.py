from dataclasses import dataclass
from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt
from utils import median_distance, get_left_right_bound
from scipy.signal import find_peaks

values = np.genfromtxt("example_data/data_r.csv", delimiter=",")
values = values[1:]


@dataclass
class Peak:
    bounds: Tuple[int, int]
    position: int


def accurate_bounds_for_peak(values, peak_bound_estimate, peak_position):
    gradients = np.gradient(values, 1)
    bounds = get_left_right_bound(
        y=gradients[peak_bound_estimate[0] : peak_bound_estimate[1]],
        px=peak_position,
        left_crop=peak_bound_estimate[0],
    )

    return bounds


def collect_estimate_bounds(values, peak_positions):
    avg_peak_distance = average_peak_distances(peak_positions)
    left_bound_estimates = np.clip(peak_positions - avg_peak_distance, 0, len(values))
    righ_bound_estimates = np.clip(peak_positions + avg_peak_distance, 0, len(values))
    return list(zip(left_bound_estimates, righ_bound_estimates))


def average_peak_distances(peak_positions):
    return abs(int(median_distance(peak_positions) / 2))


def collect_accurate_bounds(values, peak_positions, estimated_bounds):
    return [
        accurate_bounds_for_peak(values, estimated_bounds[i], p)
        for i, p in enumerate(peak_positions)
    ]


def collect_peaks(values):
    peak_positions, _ = find_peaks(values, prominence=1)
    estimated_bounds = collect_estimate_bounds(values, peak_positions)
    accurate_bounds = collect_accurate_bounds(values, peak_positions, estimated_bounds)
    zipped = zip(accurate_bounds, peak_positions)
    return [Peak(bounds, p) for bounds, p in zipped]


def plot_peaks(peaks):
    for peak in peaks:
        plt.vlines(
            peak.position, ymin=-1, ymax=2, color="blue", linestyle=":", linewidth=0.75
        )
        plt.vlines(peak.bounds[0], ymin=-1, ymax=2, color="green", linewidth=0.75)
        plt.vlines(peak.bounds[1], ymin=-1, ymax=2, color="red", linewidth=0.75)


peaks = collect_peaks(values)

plt.plot(values, color="black")
plot_peaks(peaks)
plt.show()
