import matplotlib as mpl
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
import Orange


def get_y_pred(y_pred, nlabel):
    n = len(y_pred)
    rt = np.zeros((n, nlabel))
    rt[range(0, n), y_pred] = 1
    return rt


if __name__ == '__main__':
    # a = np.array([1, 0, 3])
    # b = np.zeros((a.size, a.max() + 1))
    # print(np.arange(a.size))
    # b[np.arange(a.size), a] = 1
    # print(b)
    # raw = np.array((3, 2, 2))
    # print(raw)
    # raw[0, :] = np.ones((2, 2))
    # raw[1, :] = -2 * np.ones((2, 2))
    # print(raw)

    print(type(list(np.arange(1,2))))