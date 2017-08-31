"""
.. module:: Train

Train
*************

:Description: Train

    

:Authors: bejar
    

:Version: 

:Created on: 31/08/2017 10:36 

"""
from __future__ import print_function
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.optimizers import RMSprop
import numpy as np
from keras.utils import np_utils
import numpy as np
import pickle
from collections import Counter

__author__ = 'bejar'

dpath = '/home/bejar/PycharmProjects/DLMAI/Literature/'

class taskrecode:
    code = ({}, {}, {}, {})
    authormap = {}

    def __init__(self, labels):
        """
        Generates the recoding object

        :param labels:
        """
        # author label correspondence to the other tasks
        for i in range(labels.shape[0]):
            self.authormap[labels[i, 0]] = (labels[i, 1], labels[i, 2], labels[i, 3])

        # mapping string to integer for the tasks
        for i in range(labels.shape[1]):
            for j, v in enumerate(np.unique(labels[:, i])):
                self.code[i][v] = j

    def recode_labels(self, llabels, task=0):
        """
        Recodes a list of labels for a task
        :param llabels:
        :return:
        """
        labels = []
        if task == 0:
            for l in llabels:
                labels.append(self.code[0][l])
        else:
            for l in llabels:
                labels.append(self.code[task][self.authormap[l][task-1]])

        return labels

    def nlabels_task(self, task):
        """
        Returns the number of labels in the task

        :param task:
        :return:
        """
        return len(self.code[task])




if __name__ == '__main__':

    task = 1
    train_dm = np.load(dpath + 'train_data.npy')
    train_dm = np.reshape(train_dm, (train_dm.shape[0], train_dm.shape[1], 1))
    file = open(dpath + 'train_labels.pkl', 'rb')
    csv_labels = np.loadtxt(dpath + 'Test.csv', dtype='string', skiprows=1, delimiter= ',')
    recoder = taskrecode(csv_labels)

    train_l = pickle.load(file)
    train_l = recoder.recode_labels(train_l, task=task)
    print(Counter(train_l))
    train_l = np_utils.to_categorical(train_l, recoder.nlabels_task(task=task))
    print(train_dm.shape)
    print(train_l.shape)
    model = Sequential()
    model.add(LSTM(128, input_shape=(train_dm.shape[1], train_dm.shape[2]), activation='relu', return_sequences=True))
    model.add(LSTM(128))
    model.add(Dense(recoder.nlabels_task(task=task)))
    model.add(Activation('softmax'))

    optimizer = RMSprop(lr=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    model.summary()

    model.fit(train_dm, train_l, batch_size=128, epochs=10)

    test_dm = np.load(dpath + 'test_data.npy')
    test_dm = np.reshape(test_dm, (test_dm.shape[0], test_dm.shape[1], 1))
    file = open(dpath + 'test_labels.pkl', 'rb')
    test_l = pickle.load(file)
    test_l = recoder.recode_labels(test_l, task=task)
    test_l = np_utils.to_categorical(test_l, recoder.nlabels_task(task=task))


    print(model.evaluate(test_dm, test_l, verbose=0))