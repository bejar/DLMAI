"""
.. module:: GenPoetry

GenPoetry
*************

:Description: GenPoetry

    Generates a random poem given a model trained with TextGenerator.py

:Authors: bejar
    

:Version: 

:Created on: 09/10/2017 7:21 

"""

import numpy as np
import keras.models
import random
import gzip
import argparse
import sys

__author__ = 'bejar'

def sample(preds, temperature=1.0):
    """
    helper function to sample an index from a probability array

    :param preds:
    :param temperature:
    :return:
    """
    preds = np.asarray(preds).astype('float64')
    preds += 1.0e-50
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

def generate_text(seed, numlines):
    """
    Generates a number of lines (or at most numlines * 50 characters) using the given seed
    :param seed:
    :param lines:
    :return:
    """
    generated = ''
    gprinted = ''
    sentence = seed
    generated += sentence

    nlines = 0
    for i in range(numlines * 50):
        x = np.zeros((1, maxlen, len(chars)))
        for t, char in enumerate(sentence):
            x[0, t, char_indices[char]] = 1.

        preds = model.predict(x, verbose=0)[0]
        next_index = sample(preds, diversity)
        next_char = indices_char[next_index]
        generated += next_char
        print(next_char, end='', flush=True)
        sentence = sentence[1:] + next_char
        # Count the number of lines generated
        if next_char == '\n':
            nlines += 1
        if nlines > numlines:
            break

    print('\n')


def random_seed(chars, nchars):
    """
    Generates a random string
    :param nchars:
    :return:
    """
    s = ""
    for i in range(nchars):
        s += chars[random.randint(0, len(chars) - 1)]

    return s

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--diversity', help="Diversity for the samplig", default=0.4, type=float)
    parser.add_argument('--lines', help="Lines of text to generate", default=10, type=int)
    parser.add_argument('--model', help="model to use", default='textgen')
    args = parser.parse_args()

    if '.h5' not in  args.model:
        args.model+='.h5'
    model = keras.models.load_model(f'{args.model}')

    chars =  ['\n', ' ', '!', '"', "'", ',', '.', ':', ';', '?', 'a', 'b', 'c', 'd', 'e',
              'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
              'u', 'v', 'w', 'x', 'y', 'z', '‘']

    char_indices = dict((c, i) for i, c in enumerate(chars))
    indices_char = dict((i, c) for i, c in enumerate(chars))


    maxlen = model.layers[0].input_shape[1]
    diversity = float(args.diversity)
    seed = str(random_seed(chars, maxlen))

    print('Generating poem')
    print('*'*20)
    print()
    generate_text(seed, numlines=args.lines)
