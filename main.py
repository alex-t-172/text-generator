# -*- coding: utf-8 -*-

import rap_lyrics_data_gen
import build_model
import text_rnn_generate

def get_artist_text():
    
    artist = input('Type name of artist to work with: ')
    rap_lyrics_data_gen.generate_ohhla_text(artist)
    
    
def create_model(text_filename, save_name, max_len, n_nodes, batch_size=None, n_iter=None):
    
    text, char_indices, indices_char, chars = build_model.create_indices(text_filename)
    build_model.build_LSTM_model(chars, n_nodes, save_name)
    if batch_size and n_iter:
        X, y, = build_model.create_data_arrays(text, char_indices, indices_char, chars, max_len)
        build_model.train_LSTM_model(save_name, X, y, batch_size, n_iter)
    
    
def train_model(text_filename, model_name, max_len, batch_size, n_iter):
    
    text, char_indices, indices_char, chars = build_model.create_indices(text_filename)
    X, y, = build_model.create_data_arrays(text, char_indices, indices_char, chars, max_len)
    build_model.train_LSTM_model(model_name, X, y, batch_size, n_iter)


def text_gen_run(text_filename, save_name):
    
    text, char_indices, indices_char, chars = build_model.create_indices(text_filename)
    text_rnn_generate.generate_text(save_name, chars, char_indices, indices_char)
