import numpy as np
import matplotlib.pyplot as plt

from peakseries import peak_series
from plotting import plot_peaks


values = np.genfromtxt("example_data/data_r.csv", delimiter=",")
values = values[1:]  # We're throwing away the first value, because ...

peaks = peak_series(values)

plt.plot(values, color="black")
plot_peaks(peaks)
plt.show()
