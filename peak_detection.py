from typing import List
import numpy as np
import matplotlib.pyplot as plt

from peakseries import BoundedPeak, peak_series


def plot_peaks(peaks: List[BoundedPeak]) -> None:
    for peak in peaks:
        plt.vlines(
            peak.position, ymin=-1, ymax=2, color="blue", linestyle=":", linewidth=0.75
        )
        plt.vlines(peak.bounds[0], ymin=-1, ymax=2, color="green", linewidth=0.75)
        plt.vlines(peak.bounds[1], ymin=-1, ymax=2, color="red", linewidth=0.75)


values = np.genfromtxt("example_data/data_r.csv", delimiter=",")
values = values[1:]


peaks = peak_series(values)

plt.plot(values, color="black")
plot_peaks(peaks)
plt.show()
