# Reference: https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html

from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SOS_token = 0
EOS_token = 1

class Lang:
    def __init__ (self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {SOS_token: "SOS", EOS_token: "EOS"}
        self.n_words = 2  # Count SOS and EOS

    def addSentence (self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord (self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s

def readLangs(lang1, lang2, reverse=False):
    print("Reading lines...")

    # Read the file and split into lines
    lines = open('data/%s-%s.txt' % (lang1, lang2), encoding='utf-8').\
        read().strip().split('\n')

    # Split every line into pairs and normalize
    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]

    # Reverse pairs, make Lang instances
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    return input_lang, output_lang, pairs

MAX_LENGTH = 10

eng_prefixes = (
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s",
    "you are", "you re ",
    "we are", "we re ",
    "they are", "they re "
)

def filterPair(p):
    return len(p[0].split(' ')) < MAX_LENGTH and \
        len(p[1].split(' ')) < MAX_LENGTH and \
        p[1].startswith(eng_prefixes)

def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]

def prepareData(lang1, lang2, reverse=False):
    input_lang, output_lang, pairs = readLangs(lang1, lang2, reverse)
    print("Read %s sentence pairs" % len(pairs))
    pairs = filterPairs(pairs)
    print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counting words...")
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    return input_lang, output_lang, pairs

input_lang, output_lang, pairs = prepareData('eng', 'fra', True)
print(random.choice(pairs))

def sentence_to_indices (sentence, lang):
    """
    @param sentence (string)
    @param lang (Lang)
    @return (array(seq_len, 1))
    """
    indices = [lang.word2index[word] for word in sentence.split(' ')]
    indices.append(EOS_token)
    return torch.tensor(indices, dtype=torch.long, device=device).unsqueeze(dim=1)

inputs = [sentence_to_indices(pairs[i][0], input_lang) for i in range(len(pairs))]
targets = [sentence_to_indices(pairs[i][1], output_lang) for i in range(len(pairs))]

################################################################################

dropout_prob = 0.1
seq_len_max = MAX_LENGTH
teacher_forcing_ratio = 0.5

class EncoderRNN (nn.Module):
    def __init__ (self, dict_size, embedding_size, hidden_size):
        super(EncoderRNN, self).__init__()

        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(dict_size, embedding_size)
        self.gru = nn.GRU(embedding_size, hidden_size)

        self.optimizer = optim.Adam(self.parameters(), lr=0.01)

    def forward (self, input):
        """
        @param input (tensor(seq_len, 1, dtype=long))
        @return output (tensor(seq_len, 1, hidden_size))
        """
        self.hidden = torch.zeros(1, 1, self.hidden_size, device=device)
        embedded = self.embedding(input)  # embedded (tensor(seq_len, 1, embedding_size))
        output, self.hidden = self.gru(embedded, self.hidden)  # output (tensor(seq_len, 1, hidden_size))
        return output

class AttnDecoderRNN (nn.Module):
    def __init__ (self, dict_size, embedding_size, hidden_size, combined_size):
        super(AttnDecoderRNN, self).__init__()

        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(dict_size, embedding_size)
        self.attn = nn.Linear(embedding_size + hidden_size, seq_len_max)
        self.attn_combine = nn.Linear(embedding_size + hidden_size, combined_size)
        self.dropout = nn.Dropout(dropout_prob)
        self.gru = nn.GRU(combined_size, hidden_size)
        self.out = nn.Linear(hidden_size, dict_size)

        self.optimizer = optim.Adam(self.parameters(), lr=0.01)
        self.loss_func = nn.NLLLoss()

    def forward (self, input, encoder_output, prev_hidden):
        """
        @param input (tensor(1, 1, dtype=long))
        @param encoder_output (tensor(seq_len, 1, hidden_size))
        @param prev_hidden (tensor(1, 1, hidden_size))
        @return output (tensor(1, dict_size))
        """
        self.hidden = prev_hidden
        embedded = self.dropout(self.embedding(input))  # embedded (tensor(1, 1, embedding_size))
        attn_weights = F.softmax(self.attn(torch.cat((embedded.squeeze(dim=0), self.hidden.squeeze(dim=0)), dim=1)), dim=1)  # attn_weights (tensor(1, seq_len_max))
        attn_applied = torch.mm(attn_weights[:,:encoder_output.size(0)], encoder_output.squeeze(dim=1))  # attn_applied (tensor(1, hidden_size))
        output = torch.cat((embedded.squeeze(dim=0), attn_applied), dim=1)  # output (tensor(1, embedding_size + hidden_size))
        output = F.relu(self.attn_combine(output).unsqueeze(dim=0))  # output (tensor(1, 1, combined_size))
        output, self.hidden = self.gru(output, self.hidden)  # output (tensor(1, 1, hidden_size))
        output = F.log_softmax(self.out(output.squeeze(dim=0)), dim=1)  # output (tensor(1, dict_size))
        return output

def train (inputs, targets, encoder, decoder, epochs):
    """
    @param input ([array(seq_len, 1)])
    @param target ([array(seq_len, 1)])
    """
    examples = len(inputs)
    for epoch in range(epochs):
        for example in range(examples):
            input = inputs[example]
            target = targets[example]
            input_seq_len = input.size(0)
            target_seq_len = target.size(0)

            encoder.optimizer.zero_grad()
            decoder.optimizer.zero_grad()
            loss = 0

            encoder_output = encoder(input)
            decoder_input = torch.tensor([[SOS_token]], device=device)
            prev_hidden = encoder.hidden

            teacher_forcing = True if random.random() < teacher_forcing_ratio else False
            if teacher_forcing:
                for i in range(target_seq_len):
                    decoder_output = decoder(decoder_input, encoder_output, prev_hidden)
                    loss += decoder.loss_func(decoder_output, target[i])
                    decoder_input = torch.tensor([[target[i]]], device=device)
            else:
                for i in range(target_seq_len):
                    decoder_output = decoder(decoder_input, encoder_output, prev_hidden)
                    loss += decoder.loss_func(decoder_output, target[i])
                    topv, topi = decoder_output.topk(1)
                    decoder_input = topi.detach()
                    if decoder_input.item() == EOS_token:
                        break

            loss.backward()
            decoder.optimizer.step()
            encoder.optimizer.step()
            loss = loss.item() / target_seq_len
            print(loss)

encoder = EncoderRNN(input_lang.n_words, embedding_size=300, hidden_size=256).to(device)
decoder = AttnDecoderRNN(output_lang.n_words, embedding_size=300, hidden_size=256, combined_size=256).to(device)
train(inputs, targets, encoder, decoder, epochs=1)
