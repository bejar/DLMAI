"""
.. module:: WindPrediction

WindPrediction
*************

:Description: WindPrediction

:Authors: bejar
    

:Version: 

:Created on: 06/09/2017 9:47 

"""

import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM, GRU
from keras.optimizers import RMSprop, SGD
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import argparse

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
    parser = argparse.ArgumentParser()
    parser.add_argument('--verbose', help="Verbose output (enables Keras verbose output)", action='store_true', default=False)
    parser.add_argument('--gpu', help="Use LSTM/GRU grpu implementation", action='store_true', default=False)
    args = parser.parse_args()

    verbose = 1 if args.verbose else 0
    impl = 2 if args.gpu else 0

    ############################################
    # Data

    vars = {0: 'wind_speed', 1: 'air_density', 2: 'temperature', 3: 'pressure'}

    wind = np.load('Wind.npz')
    print(wind.files)
    wind = wind['90-45142']
    wind = wind[:, 0]

    scaler = StandardScaler()
    wind = scaler.fit_transform(wind.reshape(-1, 1))

    # Size of the training and size for validatio+test set (half for validation, half for test)
    datasize = 200000
    testsize = 40000

    # Length of the lag for the training window
    lag = 10

    wind_train = wind[:datasize, 0]
    train = lagged_vector(wind_train, lag=lag)
    train_x, train_y = train[:, :-1], train[:,-1]
    train_x = np.reshape(train_x, (train_x.shape[0], train_x.shape[1], 1))

    wind_test = wind[datasize:datasize+testsize, 0]
    test = lagged_vector(wind_test, lag=lag)
    half_test = int(test.shape[0]/2)

    val_x, val_y = test[:half_test, :-1], test[:half_test,-1]
    test_x = np.reshape(val_x, (val_x.shape[0], val_x.shape[1], 1))

    test_x, test_y = test[half_test:, :-1], test[half_test:,-1]
    test_x = np.reshape(test_x, (test_x.shape[0], test_x.shape[1], 1))

    print(train_x.shape, test_x.shape)

    ############################################
    # Model

    neurons = 64
    drop = 0.0
    nlayers = 3  # >= 1
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
    model.compile(loss='mean_squared_error', optimizer=optimizer, validation_data=(val_x, val_y))

    batch_size = 1000
    nepochs = 20

    model.fit(train_x, train_y,
              batch_size=batch_size,
              epochs=nepochs,
              verbose=verbose)

    ############################################
    # Results

    #train_predict = model.predict(train_x).flatten()
    test_predict = model.predict(test_x).flatten()

    score = model.evaluate(test_x, test_y,
                           batch_size=batch_size,
                           verbose=verbose)

    print('MSE= ', score)
    print ('MSE persistence =', mean_squared_error(test_y[1:], test_y[0:-1]))
    print(test_y.shape, test_predict.shape)

    if verbose:
        plt.subplot(2, 1, 1)
        plt.plot(test_predict, color='r')
        plt.plot(test_y, color='b')
        plt.subplot(2, 1, 2)
        plt.plot(test_y - test_predict, color='r')
        plt.show()
