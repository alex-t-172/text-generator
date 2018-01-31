# -*- coding: utf-8 -*-
"""
Created on Thu Aug  3 17:12:03 2017

@author: Alex.Thoma
"""
import sys
import numpy as np
#### well to start the name is depreciated
from keras.models import load_model
#### for two there's a better way to do this...

artist = input('Type name of artist to download text file for: ')
iteration = input('Type iteration of model to load: ')
text = open('{}_data.txt'.format(artist)).read().lower()

# remove square and round brackets
#text = re.sub("[\(\[].*?[\)\]]", "", text)

print('corpus length:', len(text))

chars = sorted(list(set(text)))
print('total chars:', len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

del(text)

model_name = '{}_LSTM_model{}.h5'.format(artist, iteration)
model = load_model(model_name)

writing_range = input('Please specify no of characters to generate:')

seed_string=input('Please specify a seed string:')
print ("seed string -->", seed_string)
print ('The generated text is')
sys.stdout.write(seed_string),
#x=np.zeros((1, len(seed_string), len(chars)))



for i in range(int(writing_range)):
    x=np.zeros((1, len(seed_string), len(chars)))
    for t, char in enumerate(seed_string):
        x[0, t, char_indices[char]] = 1.
    preds = model.predict(x, verbose=0)[0]
    #print (np.argmax(preds[7]))
    next_index=np.argmax(preds[len(seed_string)-1])
    
    
    #next_index=np.argmax(preds[len(seed_string)-11])
    #print (preds.shape)
    #print (preds)
    #next_index = sample(preds, 1) #diversity is 1
    next_char = indices_char[next_index]
    seed_string = seed_string + next_char
    
    #print (seed_string)
    #print ('##############')
    #if i==40:
    #    print ('####')
    sys.stdout.write(next_char)

sys.stdout.flush()    
