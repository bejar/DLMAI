"""
.. module:: WindPrediction

WindPrediction
*************

:Description: WindPrediction

:Authors: bejar
    

:Version: 

:Created on: 06/09/2017 9:47 

"""


from __future__ import print_function
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Activation, RepeatVector, TimeDistributed
from keras.layers import LSTM
from keras.optimizers import RMSprop, SGD
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

__author__ = 'bejar'

def lagged_vector(data, lag=1):
    """
    Returns a matrix with columns that are the steps of the lagged time series
    Last column is the value to predict
    :param data:
    :param lag:
    :return:
    """
    lvect = []
    for i in range(lag):
        lvect.append(data[i: -lag+i])
    lvect.append(data[lag:])
    return np.stack(lvect, axis=1)


if __name__ == '__main__':
    wind = np.load('wind.npy')
    scaler = StandardScaler()

    wind = scaler.fit_transform(wind.reshape(-1, 1))

    # wind = (wind - np.min(wind)) /(np.max(wind) - np.min(wind))

    lag = 20
    lenpred = 3
    train = lagged_vector(wind[:200000], lag=lag)
    train_x, train_y = train[:, :-lenpred], train[:,-lenpred:]
    train_x = np.reshape(train_x, (train_x.shape[0], train_x.shape[1], 1))

    test = lagged_vector(wind[200000:202000], lag=lag)
    test_x, test_y = test[:, :-lenpred], test[:,-lenpred:]
    test_x = np.reshape(test_x, (test_x.shape[0], test_x.shape[1], 1))

    # print(test_x[0,3,0], test_y[0])

    print(train_x.shape, train_y.shape, test_x.shape, test_y.shape)

    neurons = 256
    batch_size = 1000

    model = Sequential()
    model.add(LSTM(neurons, input_shape=(train_x.shape[1], 1), implementation=2, dropout=0.2))
    # model.add(LSTM(neurons, input_shape=(train_x.shape[1], 1), implementation=2, dropout=0.2, return_sequences=True))
    # model.add(LSTM(neurons, dropout=0.2, implementation=2))
    model.add(RepeatVector(lenpred))
    model.add(LSTM(neurons, dropout=0.2, implementation=2, return_sequences=True))
    model.add(TimeDistributed(Dense(lenpred)))

    # model.add(Activation('relu'))

    # optimizer = RMSprop(lr=0.001)
    # optimizer = SGD(lr=0.0001, momentum=0.95)
    optimizer = RMSprop(lr=0.000001)
    model.compile(loss='mean_squared_error', optimizer=optimizer)

    model.summary()
    nepochs = 5

    model.fit(train_x, train_y, batch_size=batch_size, epochs=nepochs)

    # train_predict = model.predict(train_x)
    test_predict = model.predict(test_x)

    print(test_predict.shape)
    #
    # score = model.evaluate(test_x, test_y, batch_size=batch_size)
    #
    # print('MSE= ', score)
    #
    # plt.subplot(1, 1, 1)
    # plt.plot(test_predict, color='r')
    # plt.plot(test_y, color='b')
    # plt.show()

