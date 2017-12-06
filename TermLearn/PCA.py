# coding=utf-8
"""
Crazy PCA
"""

from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import seaborn as sns

from sklearn import ensemble
from sklearn import metrics
from sklearn import linear_model
import data_paths as dp
import titles as tl
import datasets as dt
from matplotlib import pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn import preprocessing
from sklearn import svm
from sklearn import neighbors

from sklearn import ensemble
from sklearn import metrics
from sklearn import linear_model
import data_paths as dp
import titles as tl
import datasets as dt
from matplotlib import pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn import preprocessing
from sklearn import svm
from sklearn import neighbors

# Параметры обучения
C = 10.0
n_trees = 256

# Данные
target = u'прочность'
ns, X, Y = dt.get_data(target=target)

# Обучение
kfold = StratifiedKFold(n_splits=10, shuffle=True)
cvscores = []

it = 0
for train, test in kfold.split(X, Y):
    scaler = preprocessing.StandardScaler()

    sc_data = X.loc[train, tl.full_hard_non_cat_title]
    sc_data = pd.DataFrame(scaler.fit_transform(sc_data), index=sc_data.index, columns=sc_data.columns)
    ct_data = X.loc[train, tl.full_hard_cat_title]
    x_train = pd.concat([sc_data, ct_data], axis=1)
    y_train = Y[train].values

    sc_data = X.loc[test, tl.full_hard_non_cat_title]
    sc_data = pd.DataFrame(scaler.transform(sc_data), index=sc_data.index, columns=sc_data.columns)
    ct_data = X.loc[test, tl.full_hard_cat_title]
    x_test = pd.concat([sc_data, ct_data], axis=1)
    y_test = Y[test].values

    pca = PCA(n_components=10)
    XPCAreduced = pca.fit_transform(x_train)
    XTPCAreduced = pca.transform(x_test)
    print sum(pca.explained_variance_ratio_)
    plt.scatter(XPCAreduced[:, 0], XPCAreduced[:, 3])
    #plt.scatter(XPCAreduced, y_train)
    plt.show()

    # Модель
    #model = ensemble.GradientBoostingRegressor(n_estimators=n_trees, learning_rate=0.1, max_depth=3)
    model = svm.SVR(C=10)
    model.fit(XPCAreduced, y_train)
    #print model.coef_
    # y_pos = np.arange(len(model.coef_))
    # plt.barh(y_pos, model.coef_)
    # plt.yticks(y_pos, x_test.columns)
    # plt.show()

    # Тестирование
    predicted = model.predict(XTPCAreduced)
    scores = metrics.mean_squared_error(y_test, predicted)
    print('Error: {:.2f}'.format(scores))
    cvscores.append(scores)
    print('Corr: {:2f}\n'.format(np.corrcoef(y_test, predicted)[0][1]))

    #plt.hist([abs(i - j) for i, j in zip(predicted, y_test)])
    #plt.savefig('test_SVR/fig{}.jpg'.format(str(it)))
    x_test_sc = pd.DataFrame(
                            scaler.inverse_transform(x_test[tl.full_hard_non_cat_title]),
                            columns=tl.full_hard_non_cat_title,
                            index=x_test.index
    )
    x_test_ct = x_test[tl.full_hard_cat_title]
    x_test_sc[target] = y_test
    x_test_sc[target + u'(сеть)'] = predicted
    x_test_sc.combine_first(pd.DataFrame(ns)).dropna().to_excel('test_SVR/res{}.xlsx'.format(str(it)))
    it += 1


print("%.2f (+/- %.2f)" % (np.mean(cvscores), np.std(cvscores)))
