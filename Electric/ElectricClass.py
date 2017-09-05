"""
.. module:: Process

Process
*************

:Description: Process



:Authors: bejar


:Version:

:Created on: 31/08/2017 8:26

"""

from __future__ import print_function
import numpy as np


if __name__ == '__main__':
    train = np.loadtxt('ElectricDevices_TRAIN.csv', delimiter=',')
    print(train.shape)
    train_x =  train[:, 1:]
    train_y = train[:, 0]
