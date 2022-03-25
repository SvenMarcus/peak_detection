import numpy as np
import matplotlib.pyplot as plt
from utils import average_distance, get_left_bound, get_right_bound
from scipy.signal import find_peaks, peak_prominences

x = np.genfromtxt('data_L.csv', delimiter=',')
x = x[1:]
max_x, _ = find_peaks(x, prominence=1)
avg_distance = abs(int(average_distance(max_x)))

plt.plot(x)
for p in max_x:
    l = p - avg_distance if p - avg_distance > 0 else 0
    r = p + avg_distance if p + avg_distance < len(x) else len(x)
    rb = get_right_bound(x[l:r], x[max_x[0]], p - l)[-1] + l
    lb = get_left_bound(x[l:r], x[max_x[0]], p - l)[0] + l
    plt.vlines(rb, ymin=0, ymax=2.5, color="green")
    plt.vlines(lb, ymin=0, ymax=2.5, color="red")


plt.vlines(max_x, ymin=0, ymax=2.5, color="blue")

plt.show()
