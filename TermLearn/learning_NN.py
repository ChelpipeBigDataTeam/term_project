# coding=utf-8
"""
Обучение сетевых моделей
"""

import pandas as pd
import numpy as np
import seaborn as sns

import data_paths as dp
import titles as tl
import model as md
import datasets as dt
from matplotlib import pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn import preprocessing
from keras.callbacks import History
from model_methods import multirun_model

# Параметры обучения
batch_size = 32
epochs = 512
n_splits = 5
target = u'прочность'

is_plot = False;

it_count = 2
dataset_name = "full harding"

def plot(history):
    # summarize history for loss
    plt.plot(history.history['loss'])
    # plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


def run_nn(dataset_name, targetbatch_size, n_splits, is_plot=False,  epochs=epochs):
    # Данные
    ns, X, Y = dt.get_data(target=target, dataset_name=dataset_name)

    # Обучение
    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True)
    cvscores = []

    if dataset_name == 'full harding':
        non_cat_title = tl.full_hard_non_cat_title
        cat_title = tl.full_hard_cat_title
    elif dataset_name == 'double harding':
        non_cat_title = tl.double_hard_non_cat_title
        cat_title = tl.double_hard_cat_title
    else:
        raise ValueError

    it = 0
    for train, test in kfold.split(X, Y):
        scaler = preprocessing.StandardScaler()

        sc_data = X.loc[train, non_cat_title]
        sc_data = pd.DataFrame(scaler.fit_transform(sc_data), index=sc_data.index, columns=sc_data.columns)
        ct_data = X.loc[train, cat_title]
        x_train = pd.concat([sc_data, ct_data], axis=1)
        y_train = Y[train].values

        sc_data = X.loc[test, non_cat_title]
        sc_data = pd.DataFrame(scaler.transform(sc_data), index=sc_data.index, columns=sc_data.columns)
        ct_data = X.loc[test, cat_title]
        x_test = pd.concat([sc_data, ct_data], axis=1)
        y_test = Y[test].values

        # Модель
        model = md.get_simple_nn(X.shape[1])
        history = model.fit(x_train.values, y_train, epochs=epochs,
                            batch_size=batch_size, verbose=0)
        if is_plot:
            plot(history)
        scores = model.evaluate(x_test.values, y_test, verbose=0, batch_size=batch_size)
        print("%s: %.2f" % (model.metrics_names[1], scores[1]))
        cvscores.append(scores[1])

        predicted = model.predict(x_test.values, batch_size=1)
        print("corr: {:.2f}".format(np.corrcoef(y_test, [i[0] for i in predicted])[0][1]))

        # plt.hist([abs(i - j) for i, j in zip(predicted, y_test)])
        # plt.savefig(dp.test_data['learn_nn'] + '/fig{}.jpg'.format(str(it)))
        x_test_sc = pd.DataFrame(
            scaler.inverse_transform(x_test[non_cat_title]),
            columns=non_cat_title,
            index=x_test.index
        )
        x_test_ct = x_test[cat_title]
        x_test_sc[target] = y_test
        x_test_sc[target + u'(сеть)'] = predicted
        x_test_sc.combine_first(pd.DataFrame(ns)).dropna().to_excel(
            dp.test_data['learn_nn'] + '/res{}.xlsx'.format(str(it)))
        it += 1

    print("%.2f (+/- %.2f)" % (np.mean(cvscores), np.std(cvscores)))
    return [np.mean(cvscores), np.std(cvscores)]

# sums = np.array([0., 0.])
# for i in range(it_count):
#     sums += np.array(run_nn('full harding', batch_size, epochs, n_splits, target))
# print(sums/it_count)

multirun_model(run_nn, it_count,
               dataset_name, target,
               n_splits, is_plot)