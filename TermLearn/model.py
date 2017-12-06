# coding=utf-8
"""
Создание структур моделей сети
"""

import keras

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers.normalization import BatchNormalization


def get_simple_nn(n_input, n_output=1):
    model = Sequential()

    model.add(Dense(units=32, input_dim=n_input))
    model.add(Activation('tanh'))

    # model.add(Dense(units=8))
    # model.add(Activation('tanh'))

    model.add(Dense(units=n_output))

    model.compile(
        loss=keras.losses.mean_squared_error,
        # loss=keras.losses.mean_absolute_error,
        metrics=[keras.metrics.mean_squared_error],
        optimizer=keras.optimizers.SGD(lr=0.0001, momentum=0.9, decay=1e-6)
                  )

    return model
