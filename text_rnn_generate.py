# -*- coding: utf-8 -*-

import sys
import numpy as np
from keras.models import load_model
#import rap_lyrics_data_gen
import build_model

artist = input('Type name of artist to load model for: ')
X, y, char_indices, indices_char, chars = build_model.create_data(artist)
model = build_model.build_LSTM_model(X, y, chars)
build_model.train_LSTM_model(model, X, y, artist)

iteration = input('Type iteration of model to load: ')

model_name = '{}_LSTM_model{}.h5'.format(artist, iteration)
model = load_model(model_name)

writing_range = input('Please specify no of characters to generate:')

seed_string=input('Please specify a seed string:')
print ("seed string -->", seed_string)
print ('The generated text is')
sys.stdout.write(seed_string),



for i in range(int(writing_range)):
    x=np.zeros((1, len(seed_string), len(chars)))
    for t, char in enumerate(seed_string):
        x[0, t, char_indices[char]] = 1.
    preds = model.predict(x, verbose=0)[0]
    next_index=np.argmax(preds[len(seed_string)-1])
    
    
    next_char = indices_char[next_index]
    seed_string = seed_string + next_char

    sys.stdout.write(next_char)

sys.stdout.flush()    
