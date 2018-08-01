# Reference: https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html

from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random
random.seed(19980430)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

################################################################################

SOS_token = 0
EOS_token = 1

class Lang:
    def __init__ (self, name):
        self.name = name
        self.word2count = {}
        self.word2index = {}
        self.index2word = {SOS_token: "SOS", EOS_token: "EOS"}
        self.n_words = 2  # include SOS and EOS

    def add_sentence (self, sentence):
        for word in sentence.split(' '):
            self.add_word(word)

    def add_word (self, word):
        if word not in self.word2index:
            self.word2count[word] = 1
            self.word2index[word] = self.n_words
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

def normalize_string (s):
    def unicode_to_ascii(s):
        return ''.join(
            c for c in unicodedata.normalize('NFD', s)
            if unicodedata.category(c) != 'Mn'
        )
    s = unicode_to_ascii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s

def read_langs(lang1, lang2, reverse=False):
    lines = open('%s-%s.txt' % (lang1, lang2), encoding='utf-8').read().strip().split('\n')
    pairs = [[normalize_string(s) for s in line.split('\t')] for line in lines]

    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    return input_lang, output_lang, pairs

def load_data (lang1, lang2):
    print("Reading data...")
    input_lang, output_lang, pairs = read_langs(lang1, lang2, reverse=True)
    print("Read %s sentence pairs" % len(pairs))
    random.shuffle(pairs)
    for pair in pairs:
        input_lang.add_sentence(pair[0])
        output_lang.add_sentence(pair[1])
    print("Total words: (%s %s) (%s %s)" % (input_lang.name, input_lang.n_words, output_lang.name, output_lang.n_words))
    return input_lang, output_lang, pairs

def sentence_to_indices (sentence, lang):
    """
    @param sentence (string)
    @param lang (Lang)
    @return indices (array(seq_len, 1))
    """
    indices = [lang.word2index[word] for word in sentence.split(' ')]
    indices.append(EOS_token)
    return torch.tensor(indices, dtype=torch.long, device=device).unsqueeze(dim=1)

def indices_to_sentence (indices, lang):
    """
    @param indices (array(seq_len, 1))
    @param lang (Lang)
    @return sentence (string)
    """
    sentence = ''.join([(lang.index2word[indices[i,0].item()] + ' ') for i in range(indices.size(0))])
    return sentence

input_lang, output_lang, pairs = load_data('en', 'fr')
pairs_train, pairs_dev = pairs[:-1000], pairs[-1000:]
inputs_train = [sentence_to_indices(pairs[i][0], input_lang) for i in range(len(pairs_train))]
targets_train = [sentence_to_indices(pairs[i][1], output_lang) for i in range(len(pairs_train))]
inputs_dev = [sentence_to_indices(pairs[i][0], input_lang) for i in range(len(pairs_dev))]
targets_dev = [sentence_to_indices(pairs[i][1], output_lang) for i in range(len(pairs_dev))]

################################################################################

epochs = 10
embedding_size = 300
hidden_size = 256
learning_rate = 0.0001
teacher_forcing_ratio = 0.5

print_every = 1000

class EncoderGru (nn.Module):
    def __init__ (self, dict_size, embedding_size, hidden_size):
        super(EncoderGru, self).__init__()

        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(dict_size, embedding_size)
        self.gru = nn.GRU(embedding_size, hidden_size)

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward (self, input):
        """
        @param input    (seq_len, batch_size) dtype=long
        @return output  (seq_len, batch_size, hidden_size)
        """
        self.hidden = torch.zeros(1, 1, self.hidden_size, device=device)
        embedded = self.embedding(input)  # embedded (tensor(seq_len, batch_size, embedding_size))
        output, self.hidden = self.gru(embedded, self.hidden)  # output (tensor(seq_len, batch_size, hidden_size))
        return output

class DecoderGru (nn.Module):
    def __init__ (self, dict_size, embedding_size, hidden_size):
        super(DecoderGru, self).__init__()

        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(dict_size, embedding_size)
        self.gru = nn.GRU(embedding_size, hidden_size)
        self.out = nn.Linear(hidden_size, dict_size)

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.loss_func = nn.NLLLoss()

    def forward (self, input, prev_hidden, encoder_output=None):
        """
        @param input            (1, batch_size) dtype=long
        @param prev_hidden      (1, batch_size, hidden_size)
        @param encoder_output   (seq_len, batch_size, hidden_size)
        @return output          (1, batch_size, dict_size)
        @return self.hidden     (1, batch_size, hidden_size)
        """
        embedded = self.embedding(input)  # embedded (tensor(1, batch_size, embedding_size))
        _, self.hidden = self.gru(embedded, prev_hidden)
        output = F.log_softmax(self.out(self.hidden), dim=2)
        return output, self.hidden

class DecoderGruAttn (nn.Module):
    def __init__ (self, dict_size, embedding_size, hidden_size):
        super(DecoderGruAttn, self).__init__()

        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(dict_size, embedding_size)
        self.gru = nn.GRU(embedding_size, hidden_size)
        self.align = nn.Linear(hidden_size, hidden_size)
        self.attn = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, dict_size)

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.loss_func = nn.NLLLoss()

    def forward (self, input, prev_hidden, encoder_output):
        """
        @param input            (1, batch_size) dtype=long
        @param prev_hidden      (1, batch_size, hidden_size)
        @param encoder_output   (seq_len, batch_size, hidden_size)
        @return output          (1, batch_size, dict_size)
        @return self.hidden     (1, batch_size, hidden_size)
        """
        embedded = self.embedding(input)  # embedded (tensor(1, batch_size, embedding_size))
        _, self.hidden = self.gru(embedded, prev_hidden)
        attn_scores = torch.bmm(self.hidden.t(), self.align(encoder_output).t().transpose(1,2))  # attn_weights (batch_size, 1, seq_len)
        attn_weights = F.softmax(attn_scores, dim=2)  # attn_weights (batch_size, 1, seq_len)
        context = torch.bmm(attn_weights, encoder_output.t()).t()  # context (1, batch_size, hidden_size)
        attnal = F.tanh(self.attn(torch.cat((context, self.hidden), dim=2)))  # attentional (1, batch_size, hidden_size)
        output = F.log_softmax(self.out(attnal), dim=2)  # output (1, batch_size, dict_size)
        return output, self.hidden

# class DecoderGruLocalAttn (nn.Module):

def train (input, target, encoder, decoder):
    input_seq_len = input.size(0)
    target_seq_len = target.size(0)

    encoder.optimizer.zero_grad()
    decoder.optimizer.zero_grad()
    loss = 0.0

    encoder_output = encoder(input)
    decoder_input = torch.tensor([[SOS_token]], device=device)
    prev_hidden = encoder.hidden

    teacher_forcing = True if random.random() < teacher_forcing_ratio else False
    if teacher_forcing:
        for i in range(target_seq_len):
            decoder_output, prev_hidden = decoder(decoder_input, prev_hidden, encoder_output)
            loss += decoder.loss_func(decoder_output.squeeze(0), target[i])
            decoder_input = torch.tensor([[target[i]]], device=device)
    else:
        for i in range(target_seq_len):
            decoder_output, prev_hidden = decoder(decoder_input, prev_hidden, encoder_output)
            loss += decoder.loss_func(decoder_output.squeeze(0), target[i])
            topv, topi = decoder_output.topk(1)
            decoder_input = torch.tensor([[topi.detach().item()]], device=device)
            if decoder_input.item() == EOS_token:
                break

    loss.backward()
    decoder.optimizer.step()
    encoder.optimizer.step()
    loss = loss.item() / target_seq_len
    return loss

def train_all (inputs, targets, encoder, decoder, epochs):
    """
    @param inputs ([array(seq_len, 1) dtype=long])
    @param targets ([array(seq_len, 1) dtype=long])
    """
    examples = len(inputs)
    for epoch in range(epochs):
        print('Epoch %s' % epoch)
        loss_sum = 0.0
        for example in range(examples):
            input = inputs[example]
            target = targets[example]
            loss = train(input, target, encoder, decoder)
            loss_sum += loss

            if example % print_every == print_every - 1:
                loss_avg = loss_sum / print_every
                print('Loss %s' % loss_avg)
                loss_sum = 0.0
        torch.save(encoder, 'model/encoder-epoch' + str(epoch))
        torch.save(decoder, 'model/decoder-epoch' + str(epoch))

def eval (input, target, encoder, decoder):
    output = []
    with torch.no_grad():
        input_seq_len = input.size(0)
        target_seq_len = target.size(0)

        encoder_output = encoder(input)
        decoder_input = torch.tensor([[SOS_token]], device=device)
        prev_hidden = encoder.hidden

        for i in range(20):
            decoder_output, prev_hidden = decoder(decoder_input, prev_hidden, encoder_output)
            topv, topi = decoder_output.topk(1)
            decoder_input = torch.tensor([[topi.detach().item()]], device=device)
            output.append([topi.detach().item()])
            if decoder_input.item() == EOS_token:
                break
    output = torch.tensor(output, dtype=torch.long)
    return output

def eval_some (inputs, targets, encoder, decoder):
    for i in range(len(inputs)):
        input = inputs[i]
        target = targets[i]
        output = eval(input, target, encoder, decoder)
        print('< ' + indices_to_sentence(input, input_lang))
        print('= ' + indices_to_sentence(target, output_lang))
        print('> ' + indices_to_sentence(output, output_lang))
        print('')

encoder = EncoderGru(input_lang.n_words, embedding_size=embedding_size, hidden_size=hidden_size).to(device)
decoder = DecoderGruAttn(output_lang.n_words, embedding_size=embedding_size, hidden_size=hidden_size).to(device)
train_all(inputs_train, targets_train, encoder, decoder, epochs=epochs)
eval_some(inputs_dev, targets_dev, encoder, decoder)
