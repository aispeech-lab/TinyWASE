# coding=utf8
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack


class rnn_encoder(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, dropout, bidirectional):
        super(rnn_encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                           batch_first=True, dropout=dropout, bidirectional=bidirectional)

    def forward(self, input, lengths):
        self.rnn.flatten_parameters()
        embs = pack(input, list(map(int, lengths)),
                    batch_first=True)  # steps * bs * d
        outputs, (h, c) = self.rnn(embs)
        outputs = unpack(outputs)[0]
        outputs = outputs.transpose(0, 1)  # steps * bs * d -> bs * steps * d
        if self.bidirectional:
            batch_size = h.size(1)
            h = h.transpose(0, 1).contiguous().view(
                batch_size, -1, 2 * self.hidden_size)
            c = c.transpose(0, 1).contiguous().view(
                batch_size, -1, 2 * self.hidden_size)
            # 2 * (num_layers * bs * (2 * hidden_size))
            state = (h.transpose(0, 1), c.transpose(0, 1))
            return outputs, state
        else:
            return outputs, (h, c)  # (num_layers * num_directions) * bs * d
