import numpy as np
from scipy import interpolate
from matplotlib import pyplot as plt


def average_distance(a):
    distances = []
    for i in range(len(a) - 1):
        distances.append(a[i] - a[i + 1])
    return np.median(distances)


# interpolate slice
# get f linear
# min(f_lin - f_interpol)
# intersection f linear, f interpol

def get_boundaries(y, peaks, ):
    # y = mx + peak
    x = np.arange(0, len(y))
    f = interpolate.interp1d(x, y)

    return


def compare_functions(array, peak):
    array = array[:35]
    m = np.arange(0, 5, 0.1)
    x = np.arange(0, len(array), 1)
    y_lin = lambda m_, c, x_: m_ * x_ + c
    y_lin_up = lambda m_, c, l, x_: m_ * x_ +c-m_*l
    y_inter = interpolate.interp1d(x, array)

    cumsum = []

    for m_i in m:
        y_diff = abs(np.subtract(y_lin_up(m_i, peak, len(x), x), y_inter(x)))
        cumsum_i = np.cumsum(y_diff)[-1]
        cumsum.append(cumsum_i)
        plt.plot(y_diff)
        plt.plot(y_lin_up(m_i, peak, len(x), x))
        plt.plot(y_inter(x))
        plt.show()

    val, idx = min((val, idx) for (idx, val) in enumerate(cumsum))
    m_best = m[idx]
    y_diff = np.abs(np.subtract(y_lin(m_best, peak, x), y_inter(x)))
    plt.plot(y_diff)
    plt.plot(y_lin(m_best, peak, x))
    plt.plot(y_inter(x))
    plt.show()

    # slope = m[np.argmin(cumsum)[0]]
    print(m_best)
    return



def fit_line():
    """"""
