# -*- coding: utf-8 -*-

####

#modified version of text generation example in keras; trained in a many-to-many fashion using a time distributed dense layer

####

from __future__ import print_function
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM,TimeDistributed,SimpleRNN
from keras.utils.data_utils import get_file
from time import sleep
import sys

artist = input('Type name of artist to build model for: ')
text = open('{}_data.txt'.format(artist)).read().lower()

print('corpus length:', len(text))

chars = sorted(list(set(text)))
print('total chars:', len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

# split the corpus into sequences of length=maxlen
#input is a sequence of 40 chars and target is also a sequence of 40 chars shifted by one position
#for eg: if you maxlen=3 and the text corpus is abcdefghi, your input ---> target pairs will be
# [a,b,c] --> [b,c,d], [b,c,d]--->[c,d,e]....and so on
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
        
    history=model.fit(X, y, batch_size=32, nb_epoch=1,verbose=0)    #small batch size for faster training? try 16 or 32
    sleep(0.1) # https://github.com/fchollet/keras/issues/2110


    model.save('{}_deeperLSTM_model{}.h5'.format(artist, iteration))

    sys.stdout.flush()
    print ('loss is')
    print (history.history['loss'][0])
    print (history.history)
    print()