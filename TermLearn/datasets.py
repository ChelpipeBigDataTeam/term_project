# coding=utf-8
"""
Загрузка и предобработка данных
"""

import pandas as pd
import numpy as np
import seaborn as sns

import data_paths as dp
import titles as tl

from sklearn import preprocessing


def get_data(target=[u'прочность'], dataset_name='full harding'):
    data = pd.read_excel(dp.dataset[dataset_name])
    data = data.sample(frac=1.0)
    data.index = pd.Series(range(data.shape[0]))
    if dataset_name == "full harding":
        inputs = data[tl.full_hard_non_cat_title + tl.full_hard_cat_title]
    elif dataset_name == "double harding":
        inputs = data[tl.double_hard_non_cat_title + tl.double_hard_cat_title]
    else:
        raise ValueError
    targets = data[target]

    return data[u'№ партии'], inputs, targets


if __name__ == '__main__':
    _, X, _ = get_data()
    # print(X[u'удельный расход воды'])
