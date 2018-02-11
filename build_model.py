# -*- coding: utf-8 -*-

####

#modified version of text generation example in keras; trained in a many-to-many fashion using a time distributed dense layer

####

from __future__ import print_function
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM,TimeDistributed
from keras.models import load_model
from time import sleep
import sys


def create_indices(artist):
    '''Read text from file for given artist, and generate list of all characters used in file,
    and 2 dictionaries of indices per character.'''
    
    text = open('{}_data.txt'.format(artist)).read().lower()

    print('corpus length:', len(text))

    chars = sorted(list(set(text)))
    print('total chars:', len(chars))
    char_indices = dict((c, i) for i, c in enumerate(chars))
    indices_char = dict((i, c) for i, c in enumerate(chars))
    return text, char_indices, indices_char, chars


def create_data_arrays(text, artist, char_indices, indices_char, chars):
    '''Vectorize text data for given artist, spliting the corpus into sequences of length=40'''
    
    #input is a sequence of 40 chars and target is also a sequence of 40 chars shifted by one position

    maxlen = 40
    step = 1
    sentences = []
    next_chars = []
    for i in range(0, len(text) - maxlen+1, step):
        sentences.append(text[i: i + maxlen]) #input seq is from i to i  + maxlen
        next_chars.append(text[i+1:i +1+ maxlen]) # output seq is from i+1 to i+1+maxlen

    print('nb sequences:', len(sentences))

    print('Vectorization...')
    X = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
    y = np.zeros((len(sentences),maxlen, len(chars)), dtype=np.bool) # y is also a sequence , or  a seq of 1 hot vectors
    for i, sentence in enumerate(sentences):
        for t, char in enumerate(sentence):
            X[i, t, char_indices[char]] = 1

    for i, sentence in enumerate(next_chars):
        for t, char in enumerate(sentence):
            y[i, t, char_indices[char]] = 1
            
    return X, y
    

def build_LSTM_model(chars):
    '''Build a 2 stacked LSTM model to train on the corpus.'''
    
    print('Building model...')
    model = Sequential()
    model.add(LSTM(512, input_dim=len(chars),return_sequences=True))
    model.add(LSTM(512, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(TimeDistributed(Dense(len(chars))))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam')

    print ('model is made')
    print (model.summary())
    
    return model


def train_LSTM_model(model, X, y, artist):
    '''Train stacked LSTM model on vectorized corpus data, save results at each iteration.'''
    
    for iteration in range(1, 6):
        print()
        print('-' * 50)
        print('Iteration', iteration)
        
        history=model.fit(X, y, batch_size=32, nb_epoch=1,verbose=0)    #small batch size for faster training? try 16 or 32
        sleep(0.1) # https://github.com/fchollet/keras/issues/2110


        model.save('{}_LSTM_model{}.h5'.format(artist, iteration))

        sys.stdout.flush()
        print ('training history:')
        print (history.history['loss'][0])
        print (history.history)
        print()
        

def load_train_LSTM_model(X, y, artist, starting_iteration):
    '''Load stacked LSTM model, train on vectorized corpus data, save results at each iteration.'''
    
    model_name = '{}_LSTM_model{}.h5'.format(artist, starting_iteration)
    model = load_model(model_name)
    
    for iteration in range(1+starting_iteration, 10+starting_iteration):
        print()
        print('-' * 50)
        print('Iteration', iteration)
        
        history=model.fit(X, y, batch_size=32, nb_epoch=1,verbose=0)    #small batch size for faster training? try 16 or 32
        sleep(0.1) # https://github.com/fchollet/keras/issues/2110


        model.save('{}_LSTM_model{}.h5'.format(artist, iteration))

        sys.stdout.flush()
        print ('training history:')
        print (history.history['loss'][0])
        print (history.history)
        print()