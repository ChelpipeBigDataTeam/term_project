# coding=utf-8
"""
...
"""

import json
import pandas as pd

import data_paths as dp
from titles import target_to_key

from sklearn import preprocessing
from sklearn.externals import joblib
dataset_name = 'full harding'
model_type = "GB"
full_model_name = open(dp.model[dataset_name][model_type]['name'], "r").read().rstrip()

#
scaler = preprocessing.StandardScaler()

scale_data = json.load(open(dp.model[dataset_name][model_type]['scaler'], "r"))
scaler.mean_ = scale_data[0]
scaler.scale_ = scale_data[1]

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
test = pd.read_excel(dp.test_data['input'])
sc_data = test[titles_non_cat_data]
sc_data = scaler.transform(sc_data)
ct_data = test[titles_cat_data]
sc_data = pd.DataFrame(sc_data, index=ct_data.index)
inputs = sc_data.combine_first(ct_data).values

test = pd.read_excel(dp.test_data['input'])
for target_name, model_name in target_to_key.items():
    model = joblib.load(dp.model[dataset_name][model_type][model_name])
    test[target_name] = model.predict(inputs)

test.to_excel(dp.output_data['output']+'/'+full_model_name.replace(".", "") + " " + dataset_name+ " report.xlsx")
