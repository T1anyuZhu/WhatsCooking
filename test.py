import matplotlib as mpl
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
import Orange
import heapq
import pickle as pkl
import datetime
if __name__ == '__main__':
    # train_texts = pkl.load(open('data/train_text.pkl', 'rb'))
    # print(train_texts)
    now = datetime.datetime.now()
    str = now.strftime('%Y-%m-%d_%H-%M-%S')
    print(str)