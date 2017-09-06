"""
.. module:: WindPrediction

WindPrediction
*************

:Description: WindPrediction

    

:Authors: bejar
    

:Version: 

:Created on: 06/09/2017 9:47 

"""

__author__ = 'bejar'

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.optimizers import RMSprop, SGD


# frame a sequence as a supervised learning problem
def timeseries_to_supervised(data, lag=1):
	df = pd.DataFrame(data)
	columns = [df.shift(i) for i in range(lag, 0, -1)]
	columns.append(df)
	df = pd.concat(columns, axis=1)
	df.fillna(0, inplace=True)
	return df


if __name__ == '__main__':
    wind = np.load('wind.npy')


    train = wind[0:1000]
    test = wind[1000:1100]
    X, y = train[:, 0:-1], train[:, -1]
    X = X.reshape(X.shape[0], X.shape[1], 1)

    print(X.shape)

    neurons = 256
    batch_size = 256

    model = Sequential()
    model.add(LSTM(neurons, batch_input_shape=(batch_size, X.shape[1], X.shape[2]), stateful=True))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')

    nb_epoch = 50

    for i in range(nb_epoch):
        model.fit(X, y, epochs=1, batch_size=batch_size, verbose=0, shuffle=False)
        model.reset_states()