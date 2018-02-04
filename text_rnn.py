# -*- coding: utf-8 -*-

####

#modified version of text generation example in keras; trained in a many-to-many fashion using a time distributed dense layer

####

from __future__ import print_function
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM,TimeDistributed,SimpleRNN
from keras.utils.data_utils import get_file
import numpy as np
from time import sleep
import random
import sys
import os
import re
import pickle

artist = input('Type name of artist to build model for: ')

# build the model: 2 stacked LSTM
print('Build model...')
model = Sequential()
model.add(LSTM(512, input_dim=69,return_sequences=True))
model.add(LSTM(512, return_sequences=True)) #- original
model.add(LSTM(512, return_sequences=True))
model.add(Dropout(0.2))
model.add(TimeDistributed(Dense(69)))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam')

print ('model is made')

# train the model


print (model.summary())


for iteration in range(1, 6):
    print()
    print('-' * 50)
    print('Iteration', iteration)
    
        
    Xfilename = '{}_X.pickle'.format(artist)
    yfilename = '{}_y.pickle'.format(artist)
    
    with open(Xfilename, 'rb') as f:
        X = pickle.load(f)
    with open(yfilename, 'rb') as f:
        y = pickle.load(f)            
        
    history=model.fit(X, y, batch_size=32, nb_epoch=1,verbose=0)    #small batch size for faster training? try 16 or 32
    sleep(0.1) # https://github.com/fchollet/keras/issues/2110


    model.save('{}_deeperLSTM_model{}.h5'.format(artist, iteration))

    sys.stdout.flush()
    print ('loss is')
    print (history.history['loss'][0])
    print (history.history)
    print()
