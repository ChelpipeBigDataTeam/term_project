# coding=utf-8
"""
Обучение сетевых моделей
"""

import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn import preprocessing
from keras.callbacks import History
import data_paths as dp
import titles as tl
import model as md
import datasets as dt
from model_methods import multirun_model, get_titles_lists, get_x_y_data


# Параметры обучения
batch_size = 32
epochs = 512
n_splits = 5
target = u'прочность'
is_plot = True;
it_count = 1
dataset_name = "full harding"


def save_histogram(predicted, y_test, it):
    plt.hist([abs(i[0] - j) for i, j in zip(predicted, y_test)])
    plt.savefig(dp.test_data['learn_nn'] + '/fig{}.jpg'.format(str(it)))


def plot(history):
    # summarize history for loss
    plt.plot(history.history['loss'])
    # plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


def run_nn(dataset_name, target, n_splits, is_plot=False, batch_size=batch_size, epochs=epochs):
    # Данные
    ns, X, Y = dt.get_data(target=target, dataset_name=dataset_name)

    # Обучение
    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True)
    cvscores = []

    non_cat_title, cat_title = get_titles_lists(dataset_name)

    it = 0
    for train, test in kfold.split(X, Y):
        scaler = preprocessing.StandardScaler()
        x_train, y_train = get_x_y_data(X, Y, train, scaler, non_cat_title, cat_title)
        x_test, y_test = get_x_y_data(X, Y, test, scaler, non_cat_title, cat_title)

        # Модель
        model = md.get_simple_nn(X.shape[1])
        history = model.fit(x_train.values, y_train, epochs=epochs,
                            batch_size=batch_size, verbose=0)

        # if is_plot:
        #     plot(history)

        scores = model.evaluate(x_test.values, y_test, verbose=0, batch_size=batch_size)
        print("%s: %.2f" % (model.metrics_names[1], scores[1]))
        cvscores.append(scores[1])

        predicted = model.predict(x_test.values, batch_size=1)
        print("corr: {:.2f}".format(np.corrcoef(y_test, [i[0] for i in predicted])[0][1]))

        save_histogram(predicted, y_test, it)

        x_test_sc = pd.DataFrame(
            scaler.inverse_transform(x_test[non_cat_title]),
            columns=non_cat_title,
            index=x_test.index
        )

        x_test_sc[target] = y_test
        x_test_sc[target + u'(сеть)'] = predicted
        x_test_sc.combine_first(pd.DataFrame(ns)).dropna().to_excel(
            dp.test_data['learn_nn'] + '/res{}.xlsx'.format(str(it)))
        it += 1

    print("%.2f (+/- %.2f)" % (np.mean(cvscores), np.std(cvscores)))
    return [np.mean(cvscores), np.std(cvscores)]

multirun_model(run_nn, it_count, dataset_name, target, n_splits, is_plot)