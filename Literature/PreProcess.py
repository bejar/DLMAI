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
import os
import codecs
import re
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle

__author__ = 'bejar'


path = '/home/bejar/PycharmProjects/DLMAI/Literature/Data/'
dpath = '/home/bejar/PycharmProjects/DLMAI/Literature/'

def generate_files_list(path):
    """
    Generates a list of all the files inside a path (recursivelly)
    :param path:
    :return:
    """
    lfiles = []

    for lf in os.walk(path):
        if lf[2]:
            for f in lf[2]:
                lfiles.append(lf[0] + '/' + f)
    return lfiles

def generate_sentences(path, minlen=4):
    """
    Returns a list of sentences

    :param path:
    :return:
    """
    lfiles = sorted(generate_files_list(path))
    lsent = []
    llabels = []
    for f in lfiles:
        ftxt = codecs.open(f, "r", encoding='utf-8')
        lb = f.split('/')[-2].lower()
        text = ''
        for line in ftxt:
            text += line

        text = text.lower().strip()
        text = re.sub(r'[!?]', '.', text)
        text = re.sub(r'[,;()0-9\"\'\[\]:]', '', text)
        text = re.sub(r'[-_]', ' ', text)
        text = re.sub(r'[^\x00-\x7F]+',' ', text)
        ssent = text.split('.')

        llabels.extend([lb]*len(ssent))
        lsent.extend(ssent)

    sentences = []
    labels = []
    for s, l in zip(lsent, llabels):
        snt = s.split()
        if len(snt) > minlen:
            sentences.append(snt)
            labels.append(l)

    return sentences, labels

def generate_vocabulary(path):
    """

    Returns two dictionaries for converting words to indices and indices to words

    The vocabulary must be generated with the training and the test data

    :return:
    """

    sentences, _ = generate_sentences(path)

    swords = set()
    for s in sentences:
        for w in s:
            swords.add(w)

    dwords = {}

    for i, w in enumerate(swords):
        dwords[w] = i + 1

    rdwords = {}
    for w in dwords:
        rdwords[dwords[w]] = w

    return dwords, rdwords

def generate_coded_sentences(sentences, voc):
    """
    Recodes lists of words to list of indices using a word vocabulary

    :param sentences:
    :return:
    """
    lseq = []
    for s in sentences:
        seq = []
        for w in s:
            seq.append(voc[w])
        lseq.append(seq)
    return lseq

if __name__ == '__main__':

    dwords, rdwords = generate_vocabulary(path)

    sentences, labels = generate_sentences(path + 'Train')
    lseq = generate_coded_sentences(sentences, dwords)
    data = pad_sequences(lseq, truncating='post', maxlen=20, dtype='float')
    data /= float(len(dwords) +1)
    np.save(dpath + 'train_data.npy', data)
    output = open(dpath + 'train_labels.pkl', 'wb')
    pickle.dump(labels, output)

    sentences, labels = generate_sentences(path + 'Test')
    lseq = generate_coded_sentences(sentences, dwords)
    data = pad_sequences(lseq, truncating='post', maxlen=20, dtype='float')
    data /= float(len(dwords) +1)
    np.save(dpath+ 'test_data.npy', data)
    output = open(dpath + 'test_labels.pkl', 'wb')
    pickle.dump(labels, output)


