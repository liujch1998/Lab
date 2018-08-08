import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
device = torch.device('cuda:7')

import numpy as np

epochs = 1
embedding_size = 300
hidden_size = 1024
learning_rate = 1e-4
teacher_forcing_ratio = 0.5
print_every = 1000

class EncoderGru (nn.Module):

    def __init__ (self, dict_size):
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

class DecoderGru (nn.Module):

    def __init__ (self, dict_size):
        super(DecoderGru, self).__init__()

        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(dict_size, embedding_size)
        self.gru = nn.GRU(embedding_size, hidden_size)
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
        output = F.log_softmax(self.out(self.hidden), dim=2)  # output (1, batch_size, dict_size)
        return output, self.hidden

def init_model (lang):
    encoder = EncoderGru(lang.han_cnt).to(device)
    decoder = DecoderGru(lang.han_cnt).to(device)
    return encoder, decoder

def train_one (input, target, encoder, decoder):
    input_seq_len = len(input)
    target_seq_len = len(target)
    input = torch.tensor(input, dtype=torch.long, device=device).unsqueeze(1)
    target = torch.tensor(target, dtype=torch.long, device=device).unsqueeze(1)

    encoder.optimizer.zero_grad()
    decoder.optimizer.zero_grad()

    loss = 0.0
    encoder_output = encoder(input)
    decoder_input = torch.tensor([[-1]], dtype=torch.long, device=device)
    prev_hidden = encoder.hidden
    if np.random.rand() < teacher_forcing_ratio:
        for i in range(target_seq_len):
            decoder_output, prev_hidden = decoder(decoder_input, prev_hidden, encoder_output)
            loss += decoder.loss_func(decoder_output.squeeze(0), target[i])
            decoder_input = torch.tensor([[target[i,0].item()]], dtype=torch.long, device=device)
    else:
        for i in range(target_seq_len):
            decoder_output, prev_hidden = decoder(decoder_input, prev_hidden, encoder_output)
            loss += decoder.loss_func(decoder_output.squeeze(0), target[i])
            topv, topi = decoder_output.topk(1)
            decoder_input = torch.tensor([[topi.detach().item()]], dtype=torch.long, device=device)

    loss.backward()
    encoder.optimizer.step()
    decoder.optimizer.step()
    loss = loss.item() / target_seq_len
    return loss

def train_all (inputs, targets, encoder, decoder):
    n_examples = len(inputs)
    for epoch in range(epochs):
        print('Epoch %s' % epoch)
        loss_sum = 0.0
        for example in range(n_examples):
            input = inputs[example]
            target = targets[example]
            loss = train_one(input, target, encoder, decoder)
            loss_sum += loss

            if example % print_every == print_every - 1:
                loss_avg = loss_sum / print_every
                print('Loss %s' % loss_avg)
                loss_sum = 0.0
                break
        torch.save(encoder, 'model/encoder-epoch' + str(epoch))
        torch.save(decoder, 'model/decoder-epoch' + str(epoch))

def eval_one (input, target, encoder, decoder):
    output = []
    with torch.no_grad():
        input_seq_len = len(input)
        target_seq_len = len(target)
        input = torch.tensor(input, dtype=torch.long, device=device).unsqueeze(1)
        target = torch.tensor(target, dtype=torch.long, device=device).unsqueeze(1)

        encoder_output = encoder(input)
        decoder_input = torch.tensor([[-1]], dtype=torch.long, device=device)
        prev_hidden = encoder.hidden

        for i in range(target_seq_len):
            decoder_output, prev_hidden = decoder(decoder_input, prev_hidden, encoder_output)
            topv, topi = decoder_output.topk(1)
            decoder_input = torch.tensor([[topi.detach().item()]], dtype=torch.long, device=device)
            output.append(topi.detach().item())
    return output

def eval_all (inputs, targets, encoder, decoder):
    outputs = []
    for i in range(len(inputs)):
        input = inputs[i]
        target = targets[i]
        output = eval_one(input, target, encoder, decoder)
        outputs.append(output)
    return outputs