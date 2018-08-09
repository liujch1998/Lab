import os, sys
import torch

import parsing
import model

sentences = parsing.load_data(['chinese-poetry/ci/ci.song.' + str(i) + '.json' for i in range(0, 22000, 1000)])
lang = parsing.Lang(sentences)

use_epoch = sys.argv[1]
allowed = sys.argv[2:]
encoder = torch.load('model256/encoder-epoch-' + use_epoch)
decoder = torch.load('model256/decoder-epoch-' + use_epoch)

sentences_test = parsing.load_data(['test.json'])
sentences_test = [sentence for sentence in sentences_test if all([han in lang.han2index for han in sentence])]
inputs_test = [lang.sentence_to_indices(sentence) for sentence in sentences_test]
outputs_test = model.eval_all(inputs_test, encoder, decoder, lang, allowed)
for i in range(len(sentences_test)):
    print('< ' + lang.indices_to_sentence(inputs_test[i]))
    print('> ' + lang.indices_to_sentence(outputs_test[i][::-1]))
