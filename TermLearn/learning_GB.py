# coding=utf-8
"""
Обучение градиентный бустинг
"""

import numpy as np
from sklearn import ensemble
from learning_another_logic import run_model, multirun_model
from matplotlib import pyplot as plt

# Переменные для подсчета средних по запускам
it_mean_err = []
it_mean_corr = []

# Параметры обучения
target = u'прочность'
n_trees = 128
n_splits = 5
learning_rate = 0.1
max_depth = 3

dataset_name = 'full harding'

# Количество запусков
it_count = 1
# Выводить ли чертеж
is_plot = False


def plot(model, x_test, y_test):
    # #############################################################################
    # Plot training deviance

    # compute test set deviance
    test_score = np.zeros((n_trees,), dtype=np.float64)

    for i, y_pred in enumerate(model.staged_predict(x_test)):
        test_score[i] = model.loss_(y_test, y_pred)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title('Deviance')
    plt.plot(np.arange(n_trees) + 1, model.train_score_, 'b-',
             label='Training Set Deviance')
    plt.plot(np.arange(n_trees) + 1, test_score, 'r-',
             label='Test Set Deviance')
    plt.legend(loc='upper right')
    plt.xlabel('Boosting Iterations')
    plt.ylabel('Deviance')

    # #############################################################################
    # Plot feature importance
    feature_importance = model.feature_importances_
    # make importances relative to max importance
    feature_importance = 100.0 * (feature_importance / feature_importance.max())
    sorted_idx = np.argsort(feature_importance)
    pos = np.arange(sorted_idx.shape[0]) + .5
    plt.subplot(1, 2, 2)
    plt.barh(pos, feature_importance[sorted_idx], align='center')
    plt.yticks(pos, x_test.columns)
    plt.xlabel('Relative Importance')
    plt.title('Variable Importance')
    plt.show()


def run_gradient_boosting(dataset_name, target, n_splits, is_plot=True, n_trees=n_trees, learning_rate=learning_rate, max_depth=max_depth):
    model = ensemble.GradientBoostingRegressor(n_estimators=n_trees, learning_rate=learning_rate, max_depth=max_depth)
    mean, std = run_model(dataset_name, model, plot, target, n_splits, is_plot)
    return mean, std

# sums = np.array([0., 0.])
# for i in range(0, it_count):
#     sums += np.array(run_gradient_boosting(dataset_name, target, n_splits, is_plot, n_trees, learning_rate, max_depth))
# print(sums / it_count)

multirun_model(run_gradient_boosting, it_count,
               dataset_name, target, n_splits, is_plot)

