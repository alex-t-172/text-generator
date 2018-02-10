# -*- coding: utf-8 -*-

import rap_lyrics_data_gen
import build_model
import text_rnn_generate

def create_model(data_name):
    '''Vectorize, build and train model for specified dataset.'''

    text, char_indices, indices_char, chars = build_model.create_indices(data_name)
    X, y, = build_model.create_data_arrays(text, data_name, char_indices, indices_char, chars)
    model = build_model.build_LSTM_model(chars)
    build_model.train_LSTM_model(model, X, y, data_name)


def text_gen_run(data_name):
    '''Generate bulk output text for specified dataset - for pre trained model.'''
    
    text, char_indices, indices_char, chars = build_model.create_indices(data_name)
    text_rnn_generate.generate_text(data_name, chars, char_indices, indices_char)
    

def text_iter_gen_run(data_name):
    '''Generate bulk output text iteratively for specified dataset - for pre trained model.'''
    
    text, char_indices, indices_char, chars = build_model.create_indices(data_name)
    text_rnn_generate.generate_iterative_text(data_name, chars, char_indices, indices_char)
    
    
if __name__ == '__main__':
    data_name = 'bb'
    ##rap_lyrics_data_gen.generate_ohhla_text(artist)
    create_model(data_name)
    text_gen_run(data_name)
    #text_iter_gen_run(data_name)