# coding=utf-8

import pandas as pd
import numpy as np
import seaborn as sns

import data_paths as dp
import titles as tl
from matplotlib import pyplot as plt

data = pd.read_excel('../Data/prepared/prepared_data.xlsx')

data = data[data[u'примечание'] != u'до']
data = data[data[u'примечание'] != u'уд']
data = data[data[u'примечание'] != u'зи']
data = data[data[u'примечание'] != u'ви']
data = data[data[u'примечание'] != u'хк']
data = data[data[u'способ'] != u'отпуск во всех инд. ОКБ 4']
data = data[data[u'расход воды (норм.)'] != u'зак. с подстуж.']

data[u'C'] = data[u'C'].apply(lambda x: str(x).replace(',', '.')).astype(float)
data[u'Cr'] = data[u'Cr'].apply(lambda x: str(x).replace(',', '.')).astype(float)
data[u'N'] = data[u'N'].apply(lambda x: str(x).replace(',', '.')).astype(float)
data[u'темп-ра терм. (норм.)'] = data[u'темп-ра терм. (норм.)'].astype(float)
data[u'темп-ра спрейр (норм.)'] = data[u'темп-ра спрейр (норм.)'].astype(float)
data[u'темп-ра терм. (МКИ)'] = data[u'темп-ра терм. (МКИ)'].astype(float)
data[u'темп-ра спрейр (МКИ)'] = data[u'темп-ра спрейр (МКИ)'].astype(float)
data[u'темп-ра терм. (отпуск)'] = data[u'темп-ра терм. (отпуск)'].astype(float)
data[u'ОКБ (МКИ)'] = data[u'ОКБ (МКИ)'].astype(float)

data[u'предел текучести'] = data[u'предел текучести'].apply(lambda x: x/9.8 if x > 100.0 else x)
data[u'прочность'] = data[u'прочность'].apply(lambda x: x/9.8 if x > 100.0 else x)
data[u'расход воды (норм.)'] = data[u'расход воды (норм.)'].apply(lambda x: x.replace('.', '') if type(x) \
                                                                                                  == unicode else x)
data[u'расход воды (МКИ)'] = data[u'расход воды (МКИ)'].apply(lambda x: x.replace('.', '') if type(x) \
                                                                                              == unicode else x)

data[u'расход воды (норм.) (1)'] = None
data[u'расход воды (норм.) (2)'] = None
data[u'расход воды (норм.) (3)'] = None
data[u'расход воды (МКИ) (1)'] = None
data[u'расход воды (МКИ) (2)'] = None
data[u'расход воды (МКИ) (3)'] = None

data.index = pd.Series([i for i in range(data.shape[0])])
for i in range(data.shape[0]):
    try:
        r1, r2, r3 = data.loc[i, u'расход воды (норм.)'].split(u'/')
    except:
            try:
                r1, r2 = data.loc[i, u'расход воды (норм.)'].split(u'/')
                r3 = 0
            except:
                continue
    data.loc[i, u'расход воды (норм.) (1)'] = float(r1)
    data.loc[i, u'расход воды (норм.) (2)'] = float(r2)
    data.loc[i, u'расход воды (норм.) (3)'] = float(r3)
    data.loc[i, u'расход воды (норм.)'] = float(r1) + float(r2) + float(r3)

for i in range(data.shape[0]):
    try:
        r1, r2, r3 = data.loc[i, u'расход воды (МКИ)'].split(u'/')
    except:
            try:
                r1, r2 = data.loc[i, u'расход воды (МКИ)'].split(u'/')
                r3 = 0
            except:
                continue
    data.loc[i, u'расход воды (МКИ) (1)'] = float(r1)
    data.loc[i, u'расход воды (МКИ) (2)'] = float(r2)
    data.loc[i, u'расход воды (МКИ) (3)'] = float(r3)
    data.loc[i, u'расход воды (МКИ)'] = float(r1) + float(r2) + float(r3)

# data[u'расход воды (норм.)'] = data[u'расход воды (норм.)'].astype(float)
data[u'расход воды (норм.) (1)'] = data[u'расход воды (норм.) (1)'].astype(float)
data[u'расход воды (норм.) (2)'] = data[u'расход воды (норм.) (2)'].astype(float)
data[u'расход воды (норм.) (3)'] = data[u'расход воды (норм.) (3)'].astype(float)
# data[u'удельный расход воды (норм.)'] = data[u'расход воды (норм.)'] / data[u'диаметр'].astype(float)
# data[u'расход воды (МКИ)'] = data[u'расход воды (МКИ)'].astype(float)
data[u'расход воды (МКИ) (1)'] = data[u'расход воды (МКИ) (1)'].astype(float)
data[u'расход воды (МКИ) (2)'] = data[u'расход воды (МКИ) (2)'].astype(float)
data[u'расход воды (МКИ) (3)'] = data[u'расход воды (МКИ) (3)'].astype(float)
# data[u'удельный расход воды (МКИ)'] = data[u'расход воды (МКИ)'] / data[u'диаметр'].apply(lambda x: x*np.pi)

data[[u'норм_1', u'норм_2', u'норм_3', u'норм_4']] = pd.get_dummies(data[u'ОКБ (норм.)'])
data[[u'МКИ_1', u'МКИ_2', u'МКИ_3', u'МКИ_4']] = pd.get_dummies(data[u'ОКБ (МКИ)'])
data[[u'отпуск_1', u'отпуск_2', u'отпуск_3', u'отпуск_4']] = pd.get_dummies(data[u'ОКБ (отпуск)'])

data[u'длина (отпуск)'] = None
data.loc[data[u'ОКБ (отпуск)'] == 1.0, u'длина (отпуск)'] = 2.2
data.loc[data[u'ОКБ (отпуск)'] == 2.0, u'длина (отпуск)'] = 2.2
data.loc[data[u'ОКБ (отпуск)'] == 3.0, u'длина (отпуск)'] = 1.2
data.loc[data[u'ОКБ (отпуск)'] == 4.0, u'длина (отпуск)'] = 2.9

data[u'длина (закалка)'] = None
data.loc[data[u'ОКБ (норм.)'] == 1.0, u'длина (закалка)'] = 2.2
data.loc[data[u'ОКБ (норм.)'] == 2.0, u'длина (закалка)'] = 2.2
data.loc[data[u'ОКБ (норм.)'] == 3.0, u'длина (закалка)'] = 2.2
data.loc[data[u'ОКБ (норм.)'] == 4.0, u'длина (закалка)'] = 1.65
data.loc[data[u'ОКБ (МКИ)'] == 1.0, u'длина (закалка)'] = 2.2
data.loc[data[u'ОКБ (МКИ)'] == 2.0, u'длина (закалка)'] = 2.2
data.loc[data[u'ОКБ (МКИ)'] == 3.0, u'длина (закалка)'] = 2.2
data.loc[data[u'ОКБ (МКИ)'] == 4.0, u'длина (закалка)'] = 1.65
data.shape


# Полная закалка
full_hard = data[(pd.notnull(data[u'темп-ра спрейр (норм.)'])) & (pd.isnull(data[u'темп-ра терм. (МКИ)']))]
full_hard = full_hard[full_hard[u'C'] < 0.2]
full_hard[u'Mo'] = full_hard[u'Mo'].fillna(0)
full_hard[u'Ti'] = full_hard[u'Ti'].fillna(0)
full_hard[u'Nb'] = full_hard[u'Nb'].fillna(0)
full_hard = full_hard[tl.full_hard_title]
full_hard.shape


# Она же
full_hard_MKI = data[(pd.isnull(data[u'темп-ра терм. (норм.)'])) & (pd.notnull(data[u'темп-ра терм. (МКИ)']))]
full_hard_MKI = full_hard_MKI[full_hard_MKI[u'C'] < 0.2]
full_hard_MKI[u'Mo'] = full_hard_MKI[u'Mo'].fillna(0)
full_hard_MKI[u'Ti'] = full_hard_MKI[u'Ti'].fillna(0)
full_hard_MKI[u'Nb'] = full_hard_MKI[u'Nb'].fillna(0)
full_hard_MKI[u'темп-ра терм. (норм.)'] = full_hard_MKI[u'темп-ра терм. (МКИ)']
full_hard_MKI[u'темп-ра спрейр (норм.)'] = full_hard_MKI[u'темп-ра спрейр (МКИ)']
full_hard_MKI[u'расход воды (норм.)'] = full_hard_MKI[u'расход воды (МКИ)']
full_hard_MKI[u'расход воды (норм.) (1)'] = full_hard_MKI[u'расход воды (МКИ) (1)']
full_hard_MKI[u'расход воды (норм.) (2)'] = full_hard_MKI[u'расход воды (МКИ) (2)']
full_hard_MKI[u'расход воды (норм.) (3)'] = full_hard_MKI[u'расход воды (МКИ) (3)']
full_hard_MKI[[u'норм_1', u'норм_2', u'норм_3', u'норм_4']] = full_hard_MKI[[u'МКИ_1', u'МКИ_2', u'МКИ_3', u'МКИ_4']]
full_hard_MKI[u'скорость движения (норм.)'] = full_hard_MKI[u'скорость движения (МКИ)']
full_hard_MKI.shape
full_hard_MKI.to_excel('tmp.xlsx')


# Соединяем
full_hard = pd.concat([full_hard, full_hard_MKI])
full_hard = full_hard[pd.notnull(full_hard[u'предел текучести'])]
full_hard.shape


# Двухкратная закалка

notnull = pd.notnull(data[u'темп-ра терм. (норм.)'])
for title in tl.not_null_double_harding:
    notnull = notnull & pd.notnull(data[title])
double_harding = data[notnull]
double_harding = double_harding[double_harding[u'C'] < 0.2]
double_harding[u'Mo'] = double_harding[u'Mo'].fillna(0)
double_harding[u'Ti'] = double_harding[u'Ti'].fillna(0)
double_harding[u'Nb'] = double_harding[u'Nb'].fillna(0)
double_harding = double_harding[tl.double_hard_title]
double_harding.shape


# Двухкратная закалка

# double_harding = double_harding[double_harding[u'расход воды (норм.)'] != u'возд']


def f(x):
    for col in [u"предел текучести", u"прочность"]:
        if max(x[col]) - min(x[col]) >= 4.0:
            x[col] = None
        else:
            x[col] = x[col].mean()
    return x.mean()
# double_harding.to_excel('tmp.xlsx')
double_harding[ u'расход воды (норм.)'] = double_harding[ u'расход воды (норм.)'].astype(float)
double_harding[ u'расход воды (МКИ)'] = double_harding[u'расход воды (МКИ)'].astype(float)
y = double_harding.groupby([u'№ партии'])[
    u'диаметр',
    u'толщина стенки',
    u'темп-ра терм. (норм.)',
    u'темп-ра спрейр (норм.)',
    u'скорость движения (норм.)',
    u'темп-ра терм. (МКИ)',
    u'темп-ра спрейр (МКИ)',
    u'скорость движения (МКИ)',
    u'темп-ра терм. (отпуск)',
    u'темп-ра спрейр (отпуск)',
    u'скорость движения (отпуск)',
    u'предел текучести',
    u'прочность',
    # u'удлинение',
    u'HRB',
    u'C',
    u'Mn',
    u'Si',
    u'P',
    u'S',
    u'Cr',
    u'Ni',
    u'Cu',
    u'Al',
    u'V',
    #u'N',
    u'Mo',
    u'Ti',
    u'Nb',
    u'норм_1',
    u'норм_2',
    u'норм_3',
    u'норм_4',
    u'МКИ_1',
    u'МКИ_2',
    u'МКИ_3',
    u'МКИ_4',
    u'отпуск_1',
    u'отпуск_2',
    u'отпуск_3',
    u'отпуск_4',
    u'расход воды (норм.)',
    u'расход воды (норм.) (1)',
    u'расход воды (норм.) (2)',
    u'расход воды (норм.) (3)',
    u'расход воды (МКИ)',
    u'расход воды (МКИ) (1)',
    u'расход воды (МКИ) (2)',
    u'расход воды (МКИ) (3)',
    u'длина (отпуск)',
    u'длина (закалка)'
].apply(f).dropna()
y[u'удельный расход воды (норм.)'] = y[u'расход воды (норм.)'] * 1000.0 / (y[u'диаметр'] * np.pi)
y[u'удельный расход воды (МКИ)'] = y[u'расход воды (МКИ)'] * 1000.0 / (y[u'диаметр'] * np.pi)
y[u'параметр отпуска'] = (y[u'темп-ра терм. (отпуск)'] + 273.0) \
                         * (
                             20 + np.log(y[u'длина (отпуск)']) - np.log(
                                 y[u'скорость движения (отпуск)'] * 60.0)) \
                         * 1e-3
y[u'параметр закалки'] = 1.0 / (1.0 / (y[u'темп-ра терм. (МКИ)']+273.0) - 2.303 * 1.986 / 110000.0 * \
                                (np.log10(y[u'длина (закалка)']) - \
                                 np.log10(y[u'скорость движения (МКИ)'] * 60.0))) - 273.0
y = y[y[u"Mo"] < 0.1]
y = y[y[u"Mn"] < 1.0]
y.to_excel('../Data/datasets/double_harding/double_harding_data.xlsx')


sns.pairplot(y)
plt.savefig("../Data/plots/pairplot_double.jpg")


# Полная закалка

full_hard = full_hard[full_hard[u'расход воды (норм.)'] != u'воздух']


def f(x):
    for col in [u"предел текучести", u"прочность"]:
        if max(x[col]) - min(x[col]) >= 4.0:
            x[col] = None
        else:
            x[col] = x[col].mean()
    return x.mean()
y = full_hard.groupby([u'№ партии'])[
    u'диаметр',
    u'толщина стенки',
    u'темп-ра терм. (норм.)',
    u'темп-ра спрейр (норм.)',
    u'скорость движения (норм.)',
    u'темп-ра терм. (отпуск)',
    u'темп-ра спрейр (отпуск)',
    u'скорость движения (отпуск)',
    u'предел текучести',
    u'прочность',
    #u'удлинение',
    u'HRB',
    u'C',
    u'Mn',
    u'Si',
    u'P',
    u'S',
    u'Cr',
    u'Ni',
    u'Cu',
    u'Al',
    u'V',
    #u'N',
    u'Mo',
    u'Ti',
    u'Nb',
    u'норм_1',
    u'норм_2',
    u'норм_3',
    u'норм_4',
    u'отпуск_1',
    u'отпуск_2',
    u'отпуск_3',
    u'отпуск_4',
    u'расход воды (норм.)',
    u'расход воды (норм.) (1)',
    u'расход воды (норм.) (2)',
    u'расход воды (норм.) (3)',
    u'длина (отпуск)',
    u'длина (закалка)'
].apply(f).dropna()
y[u'удельный расход воды'] = y[u'расход воды (норм.)'] * 1000.0 / (y[u'диаметр'] * np.pi)
y[u'параметр отпуска'] = (y[u'темп-ра терм. (отпуск)'] + 273.0) \
                         * (
                             20 + np.log(y[u'длина (отпуск)']) - np.log(
                                 y[u'скорость движения (отпуск)'] * 60.0)) \
                         * 1e-3
y[u'параметр закалки'] = 1.0 / (1.0 / (y[u'темп-ра терм. (норм.)']+273.0) - 2.303 * 1.986 / 110000.0 * \
                                (np.log10(y[u'длина (закалка)']) - \
                                 np.log10(y[u'скорость движения (норм.)'] * 60.0))) - 273.0
y = y[y[u"Mo"] < 0.1]
y = y[y[u"Mn"] < 1.0]
y.to_excel('../Data/datasets/full_harding/full_harding_data.xlsx')


sns.pairplot(y)
plt.savefig("../Data/plots/pairplot.jpg")