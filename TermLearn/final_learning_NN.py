# coding=utf-8
"""
Обучение сетевой модели для дальнейшего использования
"""

import pandas as pd
import json
from sklearn import preprocessing

import data_paths as dp
import titles as tl
import model as md
import datasets as dt

# Параметры обучения
batch_size = 32
epochs = 512
dataset_name= "full harding"
# Данные
targets = tl.target_to_key.keys()


def save_model(dataset_name, model, target, scaler, non_cat_title, cat_title):
    # Сохранение
    path = dp.model[dataset_name]['NN']
    with open(path["name"], "w") as f:
        f.write("Neural network")

    if target in tl.target_to_key:
        model_json = model.to_json()
        json_file = open(path[tl.target_to_key[target]], "w")
        json_file.write(model_json)
        json_file.close()
        model.save_weights(path[tl.target_weights_string[target]])
    else:
        raise ValueError

    json.dump((list(scaler.mean_), list(scaler.scale_)), open(path["scaler"], "w"))
    json.dump((cat_title), open(path["titles"]['cat'], "w"))
    json.dump((non_cat_title), open(path["titles"]['non_cat'], "w"))


def learn_nn(dataset_name, targets, batch_size, epochs):
    for target in targets:
        _, X, Y = dt.get_data(target=target, dataset_name=dataset_name)

        # Модель
        model = md.get_simple_nn(X.shape[1])
        if dataset_name == 'full harding':
            non_cat_title = tl.full_hard_non_cat_title
            cat_title = tl.full_hard_cat_title
        elif dataset_name == 'double harding':
            non_cat_title = tl.double_hard_non_cat_title
            cat_title = tl.double_hard_cat_title
        else:
            raise ValueError
        # Обучение
        sc_data = X[non_cat_title]
        scaler = preprocessing.StandardScaler()
        sc_data = scaler.fit_transform(sc_data)
        sc_data = pd.DataFrame(sc_data)
        ct_data = X[cat_title]
        X = sc_data.combine_first(ct_data).values
        Y = Y.values

        model.fit(X, Y, epochs=epochs, batch_size=batch_size, verbose=1)

        save_model(dataset_name, model, target, scaler, non_cat_title, cat_title)


learn_nn(dataset_name, targets, batch_size, epochs)


# # Тестирование
# test = pd.read_excel('validation.xlsx')
# sc_data = test[tl.full_hard_non_cat_title]
# sc_data = scaler.transform(sc_data)
# sc_data = pd.DataFrame(sc_data)
# ct_data = test[tl.full_hard_cat_title]
# test = sc_data.combine_first(ct_data)
# predicted = model.predict(test.values)
# test[u'прочность (сеть)'] = predicted
# test[u'прочность'] = [59, 55, 59, 58, 55.5, 55.5, 57, 55, 56, 54, 62, 64, 52, 58, 57, 56, 59, 57, 62, 60, 64, 56, 55, 63, 56, 64, 60]
# test.to_excel("test/single/validation_res(NN).xlsx")
