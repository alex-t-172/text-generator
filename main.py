# -*- coding: utf-8 -*-

import rap_lyrics_data_gen
import build_model
import text_rnn_generate

def get_artist_text():
    '''Download song lyrics for specified artist.'''
    
    artist = input('Type name of artist to work with: ')
    rap_lyrics_data_gen.generate_ohhla_text(artist)
    
def create_model(filename, save_name):
    
    text, char_indices, indices_char, chars = build_model.create_indices(filename)
    X, y, = build_model.create_data_arrays(text, char_indices, indices_char, chars, max_len=40)
    model = build_model.build_LSTM_model(chars, n_nodes=512)
    build_model.train_LSTM_model(model, X, y, save_name)
    text_rnn_generate.generate_text(save_name, chars, char_indices, indices_char)


def text_gen_run(filename, save_name):
    '''Generate output text for specified corpus - for pre trained model.'''
    
    text, char_indices, indices_char, chars = build_model.create_indices(filename)
    text_rnn_generate.generate_text(save_name, chars, char_indices, indices_char)
