from typing import List

from matplotlib import pyplot as plt

from peakseries import BoundedPeak


def plot_peaks(peaks: List[BoundedPeak]) -> None:
    for peak in peaks:
        plt.vlines(
            peak.position, ymin=-1, ymax=2, color="blue", linestyle=":", linewidth=0.75
        )
        plt.vlines(peak.bounds[0], ymin=-1, ymax=2, color="green", linewidth=0.75)
        plt.vlines(peak.bounds[1], ymin=-1, ymax=2, color="red", linewidth=0.75)
