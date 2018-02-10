# -*- coding: utf-8 -*-

import sys
import numpy as np
from keras.models import load_model


def generate_bulk_text(artist, chars, char_indices, indices_char):
    '''Load in trained model of specified iteration, generate some output text 
    based off a seed string and save to file.'''
    
    while True:
        iteration = input('Type iteration of model to load ("exit to exit"): ')
        
        if iteration == 'exit':
            sys.exit()

        model_name = '{}_LSTM_model{}.h5'.format(artist, iteration)
        model = load_model(model_name)

        writing_range = input('Please specify no of characters to generate:')

        seed_string=input('Please specify a seed string:')

        outfile = open('{}_LSTM_model{}_{}.txt'.format(artist, iteration, seed_string), 'w')
        outfile.write(seed_string)

        for i in range(int(writing_range)):
            x=np.zeros((1, len(seed_string), len(chars)))
            for t, char in enumerate(seed_string):
                x[0, t, char_indices[char]] = 1.
            preds = model.predict(x, verbose=0)[0]
            next_index=np.argmax(preds[len(seed_string)-1])
    
    
            next_char = indices_char[next_index]
            seed_string = seed_string + next_char

            outfile.write(next_char)

        outfile.close()    


def generate_iterative_text(artist, chars, char_indices, indices_char):
    '''Load in trained model of specified iteration, generate output text 
    based off a seed strings specified every X characters, and save to file.'''
    
    iteration = input('Type iteration of model to load: ')    
    
    model_name = '{}_LSTM_model{}.h5'.format(artist, iteration)
    model = load_model(model_name)
    outfile = open('{}_LSTM_model{}.txt'.format(artist, iteration), 'w')
    
    while True:
        
        seed_string=input('Please specify a seed string (or "finish" to finish writing):')
        if seed_string == 'finish':
            break

        writing_range = input('Please specify no of characters to generate:')

        print (seed_string)
        outfile.write(seed_string)

        for i in range(int(writing_range)):
            x=np.zeros((1, len(seed_string), len(chars)))
            for t, char in enumerate(seed_string):
                x[0, t, char_indices[char]] = 1.
            preds = model.predict(x, verbose=0)[0]
            next_index=np.argmax(preds[len(seed_string)-1])
    
    
            next_char = indices_char[next_index]
            seed_string = seed_string + next_char
            
            sys.stdout.write(next_char)
            outfile.write(next_char)
            
        sys.stdout.flush() 
    outfile.close()   