import os, sys
import torch

import parsing
import model

sentences = parsing.load_data(['chinese-poetry/ci/ci.song.' + str(i) + '.json' for i in range(0, 22000, 1000)])
lang = parsing.Lang(sentences)
inputs = [lang.sentence_to_indices(sentence) for sentence in sentences]
targets = [lang.sentence_to_indices(sentence)[::-1] for sentence in sentences]
encoder, decoder = model.init_model(lang)
model.train_all(inputs, targets, encoder, decoder)
