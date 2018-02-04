# -*- coding: utf-8 -*-

import rap_lyrics_data_gen
import build_model
import text_rnn_generate

def full_run(download_flag):
    artist = input('Type name of artist to work with: ')
    
    if download_flag == 'Y':
        rap_lyrics_data_gen.generate_ohhla_text(artist)
    
    text, char_indices, indices_char, chars = build_model.create_indices(artist)
    X, y, = build_model.create_data_arrays(text, artist, char_indices, indices_char, chars)
    model = build_model.build_LSTM_model(X, y, chars)
    build_model.train_LSTM_model(model, X, y, artist)
    text_rnn_generate.generate_text(artist, chars, char_indices, indices_char)


def text_gen_run():
    artist = input('Type name of artist to work with: ')
    text, char_indices, indices_char, chars = build_model.create_indices(artist)
    text_rnn_generate.generate_text(artist, chars, char_indices, indices_char)
    

if __name__ == '__main__':
    full_run('Y')
    text_gen_run()