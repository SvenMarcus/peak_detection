import numpy as np
import matplotlib.pyplot as plt
from utils import average_distance, compare_functions
from scipy.signal import find_peaks, peak_prominences

x = np.genfromtxt('data_L.csv', delimiter=',')
x = x[1:]
max_x, _ = find_peaks(x, prominence=1)
avg_distance = abs(int(average_distance(max_x)))

compare_functions(x[0:avg_distance], x[max_x[0]])
peaks_expansion = []
peaks_f = []

for p in max_x:
    l = p - avg_distance if p - avg_distance > 0 else 0
    r = p + avg_distance if p + avg_distance < len(x) else len(x)
    slice = x[l:r]
    peak = np.asarray([p-l])
    prom = peak_prominences(slice, peak)
    plt.vlines(prom[1]+l, ymin=0, ymax=2.5, color="green")
    plt.vlines(prom[2]+l, ymin=0, ymax=2.5, color="red")

plt.plot(x)
plt.vlines(max_x, ymin=0, ymax=2.5)
plt.show()
