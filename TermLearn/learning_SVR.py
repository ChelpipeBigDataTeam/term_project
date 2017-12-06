# coding=utf-8
"""
Обучение линейная регрессия
"""

from sklearn import svm
from model_methods import run_model, multirun_model

# Переменные для подсчета средних по запускам
it_mean_err = []
it_mean_corr = []

# Параметры обучения
dataset_name = 'full harding'
target = u'прочность'
n_splits = 5
C = 200.0

# Количество запусков
it_count = 1
# Выводить ли чертеж
is_plot = False


def plot(model, x_test):
    pass


def run_SVR(dataset_name, target, n_splits, is_plot=False, C=C):
    model = svm.SVR(C=C)
    mean, std = run_model(dataset_name, 'SVR', model, plot, target, n_splits, is_plot)
    return mean, std

multirun_model(run_SVR, it_count,
               dataset_name, target, n_splits, is_plot)