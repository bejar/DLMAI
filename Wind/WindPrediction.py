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
from keras.layers import LSTM, GRU
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
    ############################################
    # Data

    vars = {0: 'wind_speed', 1: 'air_density', 2: 'temperature', 3: 'pressure'}

    wind = np.load('Wind.npz')
    print(wind.files)
    wind = wind[wind.files[0]]
    print(wind.shape)
    wind = wind[:, 0]

    scaler = StandardScaler()
    wind = scaler.fit_transform(wind.reshape(-1, 1))

    lag = 20
    wind_train = scaler.fit_transform(wind[:200000, 0].reshape(-1, 1))
    train = lagged_vector(wind_train, lag=lag)
    train_x, train_y = train[:, :-1], train[:,-1]
    train_x = np.reshape(train_x, (train_x.shape[0], train_x.shape[1], 1))

    wind_test = scaler.fit_transform(wind[200000:, 0].reshape(-1, 1))
    test = lagged_vector(wind_test, lag=lag)
    test_x, test_y = test[:, :-1], test[:,-1]
    test_x = np.reshape(test_x, (test_x.shape[0], test_x.shape[1], 1))

    print(train_x.shape, test_x.shape)

    ############################################
    # Model

    neurons = 64
    impl = 0  # 0 for CPU, 2 for GPU
    drop = 0.2
    nlayers = 2  # >= 1
    RNN = LSTM  # GRU


    model = Sequential()
    if nlayers == 1:
        model.add(RNN(neurons, input_shape=(train_x.shape[1], 1), implementation=impl, dropout=drop))
    else:
        model.add(RNN(neurons, input_shape=(train_x.shape[1], 1), implementation=impl, dropout=drop, return_sequences=True))
        for i in range(1, nlayers-1):
            model.add(RNN(neurons, dropout=drop, implementation=impl, return_sequences=True))
        model.add(RNN(neurons, dropout=drop, implementation=impl))
    model.add(Dense(1))


    ############################################
    # Training

    optimizer = RMSprop(lr=0.00001)
    model.compile(loss='mean_squared_error', optimizer=optimizer)

    batch_size = 1000
    nepochs = 50

    model.fit(train_x, train_y, batch_size=batch_size, epochs=nepochs)

    ############################################
    # Results

    train_predict = model.predict(train_x)
    test_predict = model.predict(test_x)

    score = model.evaluate(test_x, test_y, batch_size=batch_size)

    print('MSE= ', score)

    plt.subplot(1, 1, 1)
    plt.plot(test_predict, color='r')
    plt.plot(test_y, color='b')
    plt.show()

    # step prediction

    obs = np.zeros((1, lag, 1))
    pwindow = 5
    lwpred = []
    for i in range(0, 2000-(2*lag)-pwindow, pwindow):
        # copy the observations values
        for j in range(lag):
            obs[0, j, 0] = test_y[i+j]

        lpred = []
        for j in range(pwindow):
            pred = model.predict(obs)
            lpred.append(pred)
            for k in range(lag-1):
                obs[0, k, 0] = obs[0, k+1, 0]
            obs[0, -1, 0] = pred


        lwpred.append((i, np.array(lpred)))


    plt.subplot(1, 1, 1)
    plt.plot(test_y, color='b')
    for i, (_, pred) in zip(range(0, 2000, pwindow), lwpred):
        plt.plot(range(i+lag, i+lag+pwindow), np.reshape(pred,pwindow), color='r')
    plt.show()