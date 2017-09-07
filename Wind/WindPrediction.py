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
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.optimizers import RMSprop, SGD
import matplotlib.pyplot as plt

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

    lag = 8
    train = lagged_vector(wind[:100000], lag=lag)
    train_x, train_y = train[:, :-1], train[:,-1]
    train_x = np.reshape(train_x, (train_x.shape[0], train_x.shape[1], 1))

    test = lagged_vector(wind[100000:101000], lag=lag)
    test_x, test_y = test[:, :-1], test[:,-1]
    test_x = np.reshape(test_x, (test_x.shape[0], test_x.shape[1], 1))

    print(test_x[0,3,0], test_y[0])

    print(train_x.shape, test_x.shape)

    neurons = 64
    batch_size = 100

    model = Sequential()
    model.add(LSTM(neurons, input_shape=(train_x.shape[1], 1), dropout=0.2, return_sequences=True))
    model.add(LSTM(neurons, dropout=0.2, return_sequences=True))
    model.add(LSTM(neurons, dropout=0.2))
    model.add(Dense(1))
    # model.add(Activation('relu'))

    # optimizer = RMSprop(lr=0.001)
    optimizer = SGD(lr=0.0001, momentum=0.95)
    model.compile(loss='mean_squared_error', optimizer='adam')

    nepochs = 50


    model.fit(train_x, train_y, batch_size=batch_size, epochs=nepochs)

    train_predict = model.predict(train_x)
    test_predict = model.predict(test_x)

    score = model.evaluate(test_x, test_y, batch_size=batch_size)

    print('MSE= ', score)

    plt.subplot(1, 1, 1)
    plt.plot(test_predict, color='r')
    plt.plot(test_y, color='b')
    plt.show()

    # step prediction

    obs = np.zeros((1, lag))
    pwindow = 5
    lwpred = []
    for i in range(0, 1000, pwindow):
        # copy the observations values
        for j in range(lag):
            obs[0, i+j] = test_y[i+j]

        lpred = []
        for j in range(pwindow):
            pred = model.predict(obs)
            lpred.append(pred)
            obs[0, 0:-1] = obs[0, 1:]
            obs[0, -1] = pred


        lwpred.append((i, np.array(lpred)))


    plt.subplot(1, 1, 1)
    plt.plot(test_y, color='b')
    for i, (_, pred) in zip(range(0, 1000, pwindow), lwpred):
        plt.plot(range(i+lag, i+lag+pwindow), pred, color='r')
    plt.show()