# coding=utf-8
"""
Обучение иных моделей основная логика
"""
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
from sklearn import preprocessing

import data_paths as dp
import titles as tl
import datasets as dt


def get_titles_lists(dataset_name):
    if dataset_name == 'full harding':
        non_cat_title = tl.full_hard_non_cat_title
        cat_title = tl.full_hard_cat_title
    elif dataset_name == 'double harding':
        non_cat_title = tl.double_hard_non_cat_title
        cat_title = tl.double_hard_cat_title
    else:
        raise ValueError
    return non_cat_title, cat_title


def get_x_y_data(x, y, sample, scaler, cat_title, non_cat_title):
    sc_data = x.loc[sample, non_cat_title]
    sc_data = pd.DataFrame(scaler.fit_transform(sc_data), index=sc_data.index, columns=sc_data.columns)
    ct_data = x.loc[sample, cat_title]
    x_train = pd.concat([sc_data, ct_data], axis=1)
    y_train = y[sample].values
    return x_train, y_train


def multirun_model(run_model_method_name, it_count, dataset_name, target, n_splits, is_plot=False ):
    sums = np.array([0., 0.])
    for i in range(0, it_count):
        sums += np.array(
            run_model_method_name(dataset_name, target, n_splits, is_plot))
    print(sums / it_count)


def run_model(dataset_name, model, plot, target, n_splits, is_plot=False):
    ns, x, y = dt.get_data(target=target, dataset_name=dataset_name)

    # Обучение
    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True)
    cvscores = []

    cat_title, non_cat_title = get_titles_lists(dataset_name)
    it = 0
    for train, test in kfold.split(x, y):
        scaler = preprocessing.StandardScaler()
        x_train, y_train=get_x_y_data(x, y, train, scaler, cat_title, non_cat_title)
        x_test, y_test = get_x_y_data(x, y, test, scaler, cat_title, non_cat_title)
        model.fit(x_train.values, y_train)

        # Тестирование
        predicted = model.predict(x_test.values)
        scores = metrics.mean_squared_error(y_test, predicted)
        print('Error: {:.2f}'.format(scores))
        cvscores.append(scores)
        print('Corr: {:.2f}\n'.format(np.corrcoef(y_test, predicted)[0][1]))

        # plt.hist([abs(i - j) for i, j in zip(predicted, y_test)])
        # plt.savefig(dp.test_data['err_hist']+'/fig{}.jpg'.format(str(it)))

        x_test_sc = pd.DataFrame(
            scaler.inverse_transform(x_test[non_cat_title]),
            columns=non_cat_title,
            index=x_test.index
        )

        x_test_sc[target] = y_test
        x_test_sc[target + u'(сеть)'] = predicted
        x_test_sc.combine_first(pd.DataFrame(ns)).dropna().to_excel(dp.results['svr'] + '/res{}.xlsx'.format(str(it)))
        it += 1

        if is_plot:
            plot(model, x_test, y_test)

    print("%.2f (+/- %.2f)" % (np.mean(cvscores), np.std(cvscores)))
    return [np.mean(cvscores), np.std(cvscores)]