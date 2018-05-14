# -*- coding: utf-8 -*-

import sys
import numpy as np
from keras.models import load_model


def generate_text(model_name, chars, char_indices, indices_char):
    '''Load in trained model of specified iteration, generate some output text based off a seed string.'''

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
