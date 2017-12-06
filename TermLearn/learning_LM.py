# coding=utf-8
"""
Обучение линейная регрессия
"""

import numpy as np
from sklearn import linear_model
from matplotlib import pyplot as plt

from model_methods import run_model, multirun_model

# Переменные для подсчета средних по запускам
it_mean_err = []
it_mean_corr = []

# Параметры обучения
target = u'прочность'
n_splits = 5
dataset_name = "full harding"
# Количество запусков
it_count = 1
# Выводить ли чертеж
is_plot = False


def plot(model, x_test, y_test):
    y_pos = np.arange(len(model.coef_))
    plt.barh(y_pos, model.coef_)
    plt.yticks(y_pos, x_test.columns)
    plt.show()


def run_linear_regression(dataset_name, target, n_splits, is_plot=True):
   model = linear_model.LinearRegression()
   mean, std = run_model(dataset_name, 'LM', model, plot, target, n_splits, is_plot)
   return mean, std

multirun_model(run_linear_regression, it_count,
               dataset_name, target,
               n_splits, is_plot)
