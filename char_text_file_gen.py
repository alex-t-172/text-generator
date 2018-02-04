# -*- coding: utf-8 -*-

####

#modified version of text generation example in keras; trained in a many-to-many fashion using a time distributed dense layer

####

import numpy as np
import os
import pickle

artist = input('Type name of artist to download text file for: ')
text = open('{}_data.txt'.format(artist)).read().lower()

# remove square and round brackets
#text = re.sub("[\(\[].*?[\)\]]", "", text)

print('corpus length:', len(text))

chars = sorted(list(set(text)))
print('total chars:', len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

# split the corpus into sequences of length=maxlen
#input is a sequence of 40 chars and target is also a sequence of 40 chars shifted by one position
#for eg: if you maxlen=3 and the text corpus is abcdefghi, your input ---> target pairs will be
# [a,b,c] --> [b,c,d], [b,c,d]--->[c,d,e]....and so on
maxlen = 30
step = 1
sentences = []
next_chars = []
for i in range(0, len(text) - maxlen+1, step):
    sentences.append(text[i: i + maxlen]) #input seq is from i to i  + maxlen
    next_chars.append(text[i+1:i +1+ maxlen]) # output seq is from i+1 to i+1+maxlen
    #if i<10 :
       # print (text[i: i + maxlen])
        #print(text[i+1:i +1+ maxlen])
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
    

print ('Vectorization completed')

print('Saving arrays')
Xfilename = '{}_X.pickle'.format(artist)
yfilename = '{}_y.pickle'.format(artist)
with open(Xfilename, 'wb') as f:
    pickle.dump(X, f)
with open(yfilename, 'wb') as f:
    pickle.dump(y, f)
