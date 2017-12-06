# coding=utf-8
"""
Финальное тестирование модели на отложенной выборке
"""

import json
import pandas as pd
import numpy as np

import data_paths as dp
import titles as tl
# import datasets as dt

from sklearn import preprocessing
from sklearn.externals import joblib
dataset_name = 'full harding'
model_type = "GB"

model_name = open(dp.model[dataset_name][model_type]['name'], "r").read().rstrip()

#
scaler = preprocessing.StandardScaler()
scale_data = json.load(open(dp.model[dataset_name][model_type]['scaler'], "r"))
scaler.mean_ = scale_data[0]
scaler.scale_ = scale_data[1]

#
ys_model = joblib.load(dp.model[dataset_name][model_type]['ys_model'])
tr_model = joblib.load(dp.model[dataset_name][model_type]['tr_model'])


# if dataset_name == 'full harding':
#     non_cat_title = tl.full_hard_non_cat_title
#     cat_title = tl.full_hard_cat_title
# elif dataset_name == 'double harding':
#     non_cat_title = tl.double_hard_non_cat_title
#     cat_title = tl.double_hard_cat_title
# else:
#     raise ValueError

titles_non_cat_data = json.load(open(dp.model[dataset_name][model_type]['titles']['non_cat'], "r"))
titles_cat_data = json.load(open(dp.model[dataset_name][model_type]['titles']['cat'], "r"))
#
test = pd.read_excel(dp.test_data['test'])
sc_data = test[titles_non_cat_data]
sc_data = scaler.transform(sc_data)
ct_data = test[titles_cat_data]
sc_data = pd.DataFrame(sc_data, index=ct_data.index)
test = sc_data.combine_first(ct_data)
ys_predicted = ys_model.predict(test.values)
tr_predicted = tr_model.predict(test.values)

# Тестирование
test = pd.read_excel(dp.test_data['test'])
test[u"предел текучести (предсказание)"] = ys_predicted
test[u"прочность (предсказание)"] = tr_predicted
test[u'MAE (предел текучести)'] = np.abs(test[u"предел текучести"] - test[u"предел текучести (предсказание)"])
test[u'MSE (предел текучести)'] = test[u'MAE (предел текучести)'].apply(lambda x: x*x)
test[u'MAE (прочность)'] = np.abs(test[u"прочность"] - test[u"прочность (предсказание)"])
test[u'MSE (прочность)'] = test[u'MAE (прочность)'].apply(lambda x: x*x)
test.to_excel(dp.test_data['validation_res']+'/validation_res({}) {}.xlsx'.format(model_type, dataset_name))
