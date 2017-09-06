"""
.. module:: TextGenerator

TextGenerator
*************

:Description: TextGenerator

    

:Authors: bejar
    

:Version: 

:Created on: 06/09/2017 9:26 

"""

__author__ = 'bejar'

from __future__ import print_function
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.optimizers import RMSprop, SGD
from keras.utils.data_utils import get_file
import numpy as np
import random
import sys

def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

def generate_text(seed, numlines):
    """
    Generates a number of lines (or at most 1000 characters) using the given seed
    :param seed:
    :param lines:
    :return:
    """


    print()
    print('----- diversity:', diversity)

    generated = ''
    sentence = seed
    generated += sentence
    print('----- Generating with seed: "' + sentence + '"')
    sys.stdout.write(generated)

    nlines = 0
    for i in range(1000):
        x = np.zeros((1, maxlen, len(chars)))
        for t, char in enumerate(sentence):
            x[0, t, char_indices[char]] = 1.

        preds = model.predict(x, verbose=0)[0]
        next_index = sample(preds, diversity)
        next_char = indices_char[next_index]

        generated += next_char
        sentence = sentence[1:] + next_char
        # Count the number of lines generated
        if next_char == '\n':
            nlines += 1
        sys.stdout.write(next_char)
        sys.stdout.flush()
        if nlines > numlines:
            break
    print()

if __name__ == '__main__':
    # load the text file
    path = 'poetry3.txt'
    text = open(path).read().lower().decode('ascii', 'ignore')
    print('corpus length:', len(text))

    chars = sorted(list(set(text)))
    print('total chars:', len(chars))
    char_indices = dict((c, i) for i, c in enumerate(chars))
    indices_char = dict((i, c) for i, c in enumerate(chars))

    # cut the text in semi-redundant sequences of maxlen characters
    maxlen = 80
    step = 3
    sentences = []
    next_chars = []
    for i in range(0, len(text) - maxlen, step):
        sentences.append(text[i: i + maxlen])
        next_chars.append(text[i + maxlen])
    print('nb sequences:', len(sentences))

    # Vectorizes the sequences with one hot encoding
    print('Vectorization...')
    X = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
    y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
    for i, sentence in enumerate(sentences):
        for t, char in enumerate(sentence):
            X[i, t, char_indices[char]] = 1
        y[i, char_indices[next_chars[i]]] = 1

    # build the model: a single LSTM
    # Change the implementation parameter of the LSTM to 0 for CPU and 2 for GPU
    print('Build model...')
    model = Sequential()
    model.add(LSTM(256, input_shape=(maxlen, len(chars)), implementation=2, dropout=0.2, return_sequences=True))
    model.add(LSTM(256, implementation=2, dropout=0.2))
    model.add(Dense(len(chars)))
    model.add(Activation('softmax'))

    #optimizer = RMSprop(lr=0.01)
    optimizer = SGD(lr=0.005, momentum=0.95)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer)


    # train the model, output generated text after each iteration
    for iteration in range(1, 60):
        print()
        print('-' * 50)
        print('Iteration', iteration)
        model.fit(X, y,
                  batch_size=256,
                  epochs=5)

        start_index = 0 #random.randint(0, len(text) - maxlen - 1)

        for diversity in [0.25, 0.5, 0.75]:
            generate_text(text[start_index: start_index + maxlen], 15)
