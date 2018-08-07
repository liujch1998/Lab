import json

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
device = torch.device('cuda:1')

import random
random.seed(19980430)

poems = []
for i in range(0, 22000, 1000):
    filename = 'chinese-poetry/ci/ci.song.' + str(i) + '.json'
    with open(filename, 'r') as f:
        poems += json.load(f)
puncs = [
    u'\uff0c', # ,
    u'\u3002', # .
    u'\u3001', # \
    u'\u2026', # ...
    u'\uff1a', # :
    u'\uff1f', # ?
    u'\uff01', # !
    u'\uff1b', # ;
]
purges = [
    u'\u0020', # space
    u'\u300a', # <
    u'\u300b', # >
    u'\u2fda', # a han
    u'\u30d4', # a han
    u'\u201c', # "
    u'\u201d', # "
    u'\u002e', # .
]
unks = [
    u'\u25a1', # blank square
    u'\uff3f', # _
]
'''
for poem in poems:
    paragraphs = poem[u'paragraphs']
    for paragraph in paragraphs:
        for char in paragraph:
            is_punc = any([char == punc for punc in puncs])
            is_purge = any([char == purge for purge in purges])
            is_unk = any([char == unk for unk in unks])
            is_han = (ord(char) >= 0x4E00) and (ord(char) <= 0x9FFF)
            if is_punc == False and is_purge == False and is_unk == False and is_han == False:
                print(char + ' ' + hex(ord(char)) + ' ' + paragraph)
'''
sentences = []
for poem in poems:
    paragraphs = poem[u'paragraphs']
    for paragraph in paragraphs:
        is_valid = True
        sentence = u''
        for char in paragraph:
            if char in puncs or char in purges:
                continue
            if char in unks:
                sentence += u'\u25a1'
            elif (ord(char) >= 0x4E00) and (ord(char) <= 0x9FFF):
                sentence += char
            else:
                is_valid = False
        if sentence == u'':
            is_valid = False
        if is_valid:
            sentences.append(sentence)

han_cnt = 0
han2index = {}
index2han = {}
for sentence in sentences:
    for han in sentence:
        if han not in han2index:
            index = han_cnt
            han2index[han] = index
            index2han[index] = han
            han_cnt += 1
print('Gathered %d Han characters' % han_cnt)
print('Gathered %d sentences' % len(sentences))

def sentence_to_indices (sentence):
    indices = [han2index[han] for han in sentence]
    return torch.tensor(indices, dtype=torch.long, device=device).unsqueeze(dim=1)

def indices_to_sentence (indices):
    sentence = u''.join([index2han[indices[i,0].item()] for i in range(indices.size(0))])
    return sentence

inputs_train = [sentence_to_indices(sentence) for sentence in sentences[:-10]]
targets_train = [sentence_to_indices(sentence) for sentence in sentences[:-10]]
inputs_dev = [sentence_to_indices(sentence) for sentence in sentences[-10:]]
targets_dev = [sentence_to_indices(sentence) for sentence in sentences[-10:]]

################################################################################

epochs = 1
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
        '''
        @param  input   (seq_len, batch_size) dtype=long
        @return output  (seq_len, batch_size, hidden_size)
        '''
        self.hidden = torch.zeros((1, 1, self.hidden_size), device=device)
        embedded = self.embedding(input)  # embedded (seq_len, 1, embedding_size)
        output, self.hidden = self.gru(embedded, self.hidden)  # output (seq_len, 1, hidden_size)
        return output

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
        '''
        @param  input           (1, batch_size) dtype=long
        @param  prev_hidden     (1, batch_size, hidden_size)
        @param  encoder_output  (seq_len, batch_size, hidden_size)
        @return output          (1, batch_size, dict_size)
        @return self.hidden     (1, batch_size, hidden_size)
        '''
        embedded = None
        if input.item() == -1:
            embedded = torch.zeros((1, 1, embedding_size), device=device)
        else:
            embedded = self.embedding(input)  # embedded (1, batch_size, embedding_size)
        _, self.hidden = self.gru(embedded, prev_hidden)
        attn_scores = torch.bmm(self.hidden.t(), self.align(encoder_output).t().transpose(1,2))  # attn_scores (batch_size, 1, seq_len)
        attn_weights = F.softmax(attn_scores, dim=2)  # attn_weights (batch_size, 1, seq_len)
        context = torch.bmm(attn_weights, encoder_output.t()).t()  # context (1, batch_size, hidden_size)
        attnal = F.tanh(self.attn(torch.cat((context, self.hidden), dim=2))) # attnal (1, batch_size, hidden_size)
        output = F.log_softmax(self.out(attnal), dim=2)  # output (1, batch_size, dict_size)
        return output, self.hidden

def train_one (input, target, encoder, decoder):
    input_seq_len = input.size(0)
    target_seq_len = target.size(0)

    encoder.optimizer.zero_grad()
    decoder.optimizer.zero_grad()
    loss = 0.0

    encoder_output = encoder(input)
    decoder_input = torch.tensor([[-1]], dtype=torch.long, device=device)
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

    loss.backward()
    decoder.optimizer.step()
    encoder.optimizer.step()
    loss = loss.item() / target_seq_len
    return loss

def train_all (inputs, targets, encoder, decoder, epochs):
    '''
    @param  inputs [(seq_len, 1) dtype=long]
    @param  targets [(seq_len, 1) dtype=long]
    '''
    examples = len(inputs)
    for epoch in range(epochs):
        print('Epoch %s' % epoch)
        loss_sum = 0.0
        for example in range(examples):
            input = inputs[example]
            target = targets[example]
            loss = train_one(input, target, encoder, decoder)
            loss_sum += loss

            if example % print_every == print_every - 1:
                loss_avg = loss_sum / print_every
                print('Loss %s' % loss_avg)
                loss_sum = 0.0
        torch.save(encoder, 'model/encoder-epoch' + str(epoch))
        torch.save(decoder, 'model/decoder-epoch' + str(epoch))

def eval_one (input, target, encoder, decoder):
    output = []
    with torch.no_grad():
        input_seq_len = input.size(0)
        target_seq_len = target.size(0)

        encoder_output = encoder(input)
        decoder_input = torch.tensor([[-1]], dtype=torch.long, device=device)
        prev_hidden = encoder.hidden

        for i in range(target_seq_len):
            decoder_output, prev_hidden = decoder(decoder_input, prev_hidden, encoder_output)
            topv, topi = decoder_output.topk(1)
            decoder_input = torch.tensor([[topi.detach().item()]], device=device)
            output.append([topi.detach().item()])
    output = torch.tensor(output, dtype=torch.long)
    return output

def eval_some (inputs, targets, encoder, decoder):
    for i in range(len(inputs)):
        input = inputs[i]
        target = targets[i]
        output = eval_one(input, target, encoder, decoder)
        print('< ' + indices_to_sentence(input))
        print('> ' + indices_to_sentence(output))
        print('')

encoder = EncoderGru(han_cnt, embedding_size=embedding_size, hidden_size=hidden_size).to(device)
decoder = DecoderGruAttn(han_cnt, embedding_size=embedding_size, hidden_size=hidden_size).to(device)
train_all(inputs_train, targets_train, encoder, decoder, epochs=epochs)
eval_some(inputs_train[:10], targets_train[:10], encoder, decoder)
eval_some(inputs_dev, targets_dev, encoder, decoder)
