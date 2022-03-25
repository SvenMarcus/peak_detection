import numpy as np
from scipy import interpolate
from matplotlib import pyplot as plt


def average_distance(a):
    distances = []
    for i in range(len(a) - 1):
        distances.append(a[i] - a[i + 1])
    return np.median(distances)


def get_left_bound(array, peak, p_idx):
    array = array[:p_idx]
    m = np.arange(0, 1, 0.01)
    x = np.arange(0, len(array), 1)
    y_inter = interpolate.interp1d(x, array)
    y_lin = lambda m_, c_, l_, x_: m_ * x_ + c_ - m_ * l_
    m_best = fit_line(m=m, c=peak, l=len(x), x=x, y_inter=y_inter, y_lin=y_lin)
    y_diff = np.diff(np.sign(y_lin(m_best, peak, len(x), x) - y_inter(x)))
    idx_intersection = np.argwhere(y_diff).flatten()
    return idx_intersection


def get_right_bound(array, peak, p_idx):
    array = array[p_idx:]
    m = np.arange(-0.5, 0., 0.01)
    x = np.arange(0, len(array), 1)
    y_inter = interpolate.interp1d(x, array)
    y_lin = lambda m_, c_, l_, x_: m_ * x_ + c_ - m_ * l_
    m_best = fit_line(m=m, c=peak, l=0, x=x, y_inter=y_inter, y_lin=y_lin)
    y_diff = np.diff(np.sign(y_lin(m_best, peak, 0, x) - y_inter(x)))
    idx_intersection = np.argwhere(y_diff).flatten()
    return idx_intersection


def fit_line(m, c, l, x, y_inter, y_lin):
    cumsum = []
    for m_i in m:
        y_diff = abs(np.subtract(y_lin(m_i, c, l, x), y_inter(x)))
        cumsum_i = np.cumsum(y_diff)[-1]
        cumsum.append(cumsum_i)
        #plt.plot(y_diff)
        #plt.plot(y_lin(m_i, c, l, x))
        #plt.plot(y_inter(x))
        #plt.show()

    val, idx = min((val, idx) for (idx, val) in enumerate(cumsum))
    m_best = m[idx]
    return m_best
