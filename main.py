# -*- coding: utf-8 -*-

import rap_lyrics_data_gen
import build_model
import text_rnn_generate

rap_lyrics_data_gen.generate_ohhla_text()
artist = input('Type name of artist to load model for: ')
X, y, char_indices, indices_char, chars = build_model.create_data(artist)
model = build_model.build_LSTM_model(X, y, chars)
build_model.train_LSTM_model(model, X, y, artist)
text_rnn_generate.generate_text(artist, chars, char_indices, indices_char)