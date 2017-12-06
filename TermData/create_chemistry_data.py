# coding=utf-8
"""
Создание сводных таблиц хим. анализа из малахита
"""

import pandas as pd
import numpy as np

import data_paths as dp
from titles import chemistry_title


def ch2f(x):
    if ';' in str(x):
        if '<' in str(x):
            return 0.0
        return np.mean(map(float, str(x).replace(',', '.').split(';')))
    else:
        return float(str(x).replace(',', '.'))


def create_chemistry_data():
    for key, path in dp.raw_chemistry_data.items():
        data = pd.read_excel(path)
        data = data.rename(columns=data.iloc[0, :]).loc[3:, :]
        data = data[[u'№ плавки', u'Контр. хим.анализ'] + chemistry_title]
        for element in chemistry_title:
            data[element] = data[element].apply(ch2f)
        real_data = data[data[u'Контр. хим.анализ'] == 1].drop_duplicates()
        sertif_data = data[data[u'Контр. хим.анализ'] == 0].drop_duplicates()
        real_data = real_data[pd.notnull(real_data[u'C'])]
        sertif_data = sertif_data[pd.notnull(sertif_data[u'C'])]
        real_data.to_excel(dp.clear_chemistry_data[key + '_real'])
        sertif_data.to_excel(dp.clear_chemistry_data[key + '_sertif'])
