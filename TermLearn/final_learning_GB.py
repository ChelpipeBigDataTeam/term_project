# coding=utf-8
"""
Обучение сетевой модели для дальнейшего использования
"""

import json
import pandas as pd
from sklearn import preprocessing
from sklearn import ensemble
from sklearn.externals import joblib

import data_paths as dp
import titles as tl
import datasets as dt


# Параметры обучения
n_trees = 512
learning_rate = 0.1
max_depth = 3
dataset_name = 'full harding'
# Данные
targets = tl.target_to_key.keys()


def save_model(dataset_name, model, target, scaler, non_cat_title, cat_title):
    # Сохранение
    path = dp.model[dataset_name]['GB']
    with open(path["name"], "w") as f:
        f.write(model.__doc__.split("\n")[0].rstrip())

    if target in tl.target_to_key:
        joblib.dump(model, path[tl.target_to_key[target]])
    else:
        raise ValueError

    json.dump((list(scaler.mean_), list(scaler.scale_)), open(path["scaler"], "w"))
    json.dump((cat_title), open(path["titles"]['cat'], "w"))
    json.dump((non_cat_title), open(path["titles"]['non_cat'], "w"))

def run_GB_learning(dataset_name, n_trees, learning_rate, max_depth):
    for target in targets:
        _, X, Y = dt.get_data(target=target, dataset_name=dataset_name)

        # Модель
        model = ensemble.GradientBoostingRegressor(n_estimators=n_trees, learning_rate=learning_rate, max_depth=max_depth)

        # Обучение
        if dataset_name == 'full harding':
            non_cat_title = tl.full_hard_non_cat_title
            cat_title = tl.full_hard_cat_title
        elif dataset_name == 'double harding':
            non_cat_title = tl.double_hard_non_cat_title
            cat_title = tl.double_hard_cat_title
        else:
            raise ValueError
        sc_data = X[non_cat_title]
        scaler = preprocessing.StandardScaler()
        sc_data = scaler.fit_transform(sc_data)
        sc_data = pd.DataFrame(sc_data)
        ct_data = X[cat_title]
        X = sc_data.combine_first(ct_data).values
        Y = Y.values

        model.fit(X, Y)

        save_model(dataset_name, model, target, scaler, non_cat_title, cat_title)

run_GB_learning(dataset_name, n_trees, learning_rate, max_depth)
# # # Тестирование
# test = pd.read_excel('validation.xlsx')
# sc_data = test[tl.full_hard_non_cat_title]
# sc_data = scaler.transform(sc_data)
# sc_data = pd.DataFrame(sc_data)
# ct_data = test[tl.full_hard_cat_title]
# test = sc_data.combine_first(ct_data)
# predicted = model.predict(test.values)
# test[target + u'(модель)'] = predicted
# test[target] = [59, 55, 59, 58, 55.5, 55.5, 57, 55, 56, 54, 62, 64, 52, 58, 57, 56, 59, 57, 62, 60, 64, 56, 55, 63, 56, 64, 60]
# test[u'p'] = np.abs(test[target + u'(модель)'] - test[target])
# test.to_excel("test_SVR/single/validation_res(SVR).xlsx")
