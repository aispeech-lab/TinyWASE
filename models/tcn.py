import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class cLN(nn.Module):
    def __init__(self, dimension, eps=1e-8, trainable=True):
        super(cLN, self).__init__()

        self.eps = eps
        if trainable:
            self.gain = nn.Parameter(torch.ones(1, dimension, 1))
            self.bias = nn.Parameter(torch.zeros(1, dimension, 1))
        else:
            self.gain = Variable(torch.ones(
                1, dimension, 1), requires_grad=False)
            self.bias = Variable(torch.zeros(
                1, dimension, 1), requires_grad=False)

    def forward(self, input):
        # input size: (Batch, Freq, Time)
        # cumulative mean for each time step

        batch_size = input.size(0)
        channel = input.size(1)
        time_step = input.size(2)

        step_sum = input.sum(1)  # B, T
        step_pow_sum = input.pow(2).sum(1)  # B, TTCN
        cum_sum = torch.cumsum(step_sum, dim=1)  # B, T
        cum_pow_sum = torch.cumsum(step_pow_sum, dim=1)  # B, T

        entry_cnt = np.arange(channel, channel*(time_step+1), channel)
        entry_cnt = torch.from_numpy(entry_cnt).type(input.type())
        entry_cnt = entry_cnt.view(1, -1).expand_as(cum_sum)

        cum_mean = cum_sum / entry_cnt  # B, T
        cum_var = (cum_pow_sum - 2*cum_mean*cum_sum) / \
            entry_cnt + cum_mean.pow(2)  # B, T
        cum_std = (cum_var + self.eps).sqrt()  # B, T

        cum_mean = cum_mean.unsqueeze(1)
        cum_std = cum_std.unsqueeze(1)

        x = (input - cum_mean.expand_as(input)) / cum_std.expand_as(input)
        return x * self.gain.expand_as(x).type(x.type()) + self.bias.expand_as(x).type(x.type())


def repackage_hidden(h):
    """
    Wraps hidden states in new Variables, to detach them from their history.
    """

    if type(h) == Variable:
        return Variable(h.data)
    else:
        return tuple(repackage_hidden(v) for v in h)


class MultiRNN(nn.Module):
    """
    Container module for multiple stacked RNN layers.

    args:
        rnn_type: string, select from 'RNN', 'LSTM' and 'GRU'.
        input_size: int, dimension of the input feature. The input should have shape 
                    (batch, seq_len, input_size).
        hidden_size: int, dimension of the hidden state. The corresponding output should 
                    have shape (batch, seq_len, hidden_size).
        num_layers: int, number of stacked RNN layers. Default is 1.
        bidirectional: bool, whether the RNN layers are bidirectional. Default is False.
    """

    def __init__(self, rnn_type, input_size, hidden_size, dropout=0, num_layers=1, bidirectional=False):
        super(MultiRNN, self).__init__()

        self.rnn = getattr(nn, rnn_type)(input_size, hidden_size, num_layers, dropout=dropout,
                                         batch_first=True, bidirectional=bidirectional)

        self.rnn_type = rnn_type
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_direction = int(bidirectional) + 1

    def forward(self, input):
        hidden = self.init_hidden(input.size(0))
        self.rnn.flatten_parameters()
        return self.rnn(input, hidden)

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return (Variable(weight.new(self.num_layers*self.num_direction, batch_size, self.hidden_size).zero_()),
                    Variable(weight.new(self.num_layers*self.num_direction, batch_size, self.hidden_size).zero_()))
        else:
            return Variable(weight.new(self.num_layers*self.num_direction, batch_size, self.hidden_size).zero_())


class FCLayer(nn.Module):
    """
    Container module for a fully-connected layer.

    args:
        input_size: int, dimension of the input feature. The input should have shape 
                    (batch, input_size).
        hidden_size: int, dimension of the output. The corresponding output should 
                    have shape (batch, hidden_size).
        nonlinearity: string, the nonlinearity applied to the transformation. Default is None.
    """

    def __init__(self, input_size, hidden_size, bias=True, nonlinearity=None):
        super(FCLayer, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.FC = nn.Linear(self.input_size, self.hidden_size, bias=bias)
        if nonlinearity:
            self.nonlinearity = getattr(F, nonlinearity)
        else:
            self.nonlinearity = None

        self.init_hidden()

    def forward(self, input):
        if self.nonlinearity is not None:
            return self.nonlinearity(self.FC(input))
        else:
            return self.FC(input)

    def init_hidden(self):
        initrange = 1. / np.sqrt(self.input_size * self.hidden_size)
        self.FC.weight.data.uniform_(-initrange, initrange)
        if self.bias:
            self.FC.bias.data.fill_(0)


class DepthConv1d(nn.Module):

    def __init__(self, input_channel, hidden_channel, kernel, padding, dilation=1, skip=True, causal=False):
        super(DepthConv1d, self).__init__()

        self.causal = causal
        self.skip = skip

        self.conv1d = nn.Conv1d(input_channel, hidden_channel, 1)
        if self.causal:
            self.padding = (kernel - 1) * dilation
        else:
            self.padding = padding
        self.dconv1d = nn.Conv1d(hidden_channel, hidden_channel, kernel, dilation=dilation,
                                 groups=hidden_channel,
                                 padding=self.padding)
        self.res_out = nn.Conv1d(hidden_channel, input_channel, 1)
        self.nonlinearity1 = nn.PReLU()
        self.nonlinearity2 = nn.PReLU()
        if self.causal:
            self.reg1 = cLN(hidden_channel, eps=1e-08)
            self.reg2 = cLN(hidden_channel, eps=1e-08)
        else:
            self.reg1 = nn.GroupNorm(1, hidden_channel, eps=1e-08)
            self.reg2 = nn.GroupNorm(1, hidden_channel, eps=1e-08)

        if self.skip:
            self.skip_out = nn.Conv1d(hidden_channel, input_channel, 1)

    def forward(self, input):
        output = self.reg1(self.nonlinearity1(self.conv1d(input)))
        if self.causal:
            output = self.reg2(self.nonlinearity2(
                self.dconv1d(output)[:, :, :-self.padding]))
        else:
            output = self.reg2(self.nonlinearity2(self.dconv1d(output)))
        residual = self.res_out(output)
        if self.skip:
            skip = self.skip_out(output)
            return residual, skip
        else:
            return residual


class TCN(nn.Module):
    def __init__(self, input_dim, output_dim, BN_dim, hidden_dim,
                 layer, stack, kernel=3, skip=True, win=16, stride=8,
                 causal=False, dilated=True, low_latency=5, cues="voiceprint_onset_offset"):
        super(TCN, self).__init__()
        # input is a sequence of features of shape (B, N, L)
        self.layer = layer # 8
        self.stack = stack # 3
        self.win = win
        self.stride = stride
        self.cues = cues

        print('Cues used in TCN is ', self.cues)
        if not causal:
            self.LN = nn.GroupNorm(1, input_dim, eps=1e-8)
        else:
            self.LN = cLN(input_dim, eps=1e-8)
        self.BN = nn.Conv1d(input_dim, BN_dim, 1)

        # TCN for feature extraction
        self.receptive_field = 0
        self.dilated = dilated
        self.TCN = nn.ModuleList([])
        for s in range(stack):
            for i in range(layer):
                if s * layer + i < low_latency:
                    causal = False
                else:
                    causal = True
                print('layer:', s * layer + i, 'causal:', causal)
                if self.dilated:
                    self.TCN.append(DepthConv1d(
                        BN_dim, hidden_dim, kernel, dilation=2**i, padding=2**i, skip=skip, causal=causal))
                else:
                    self.TCN.append(DepthConv1d(
                        BN_dim, hidden_dim, kernel, dilation=1, padding=1, skip=skip, causal=causal))
                if i == 0 and s == 0:
                    self.receptive_field += kernel
                else:
                    if self.dilated:
                        self.receptive_field += (kernel - 1) * 2**i
                    else:
                        self.receptive_field += (kernel - 1)
        print("Receptive field: {:3d} frames.".format(self.receptive_field))
        # output layer
        # self.output = nn.Sequential(
        #     nn.PReLU(), nn.Conv1d(BN_dim, output_dim, 1))
        self.output_act = nn.PReLU()
        self.output_conv = nn.Conv1d(BN_dim, output_dim, 1)
        self.skip = skip
        self.proj = nn.ModuleList([])
        for s in range(stack + 1):
            self.proj.append(nn.Conv1d(BN_dim, BN_dim, 1))
        # self.endpoint_dconv1d = nn.Conv1d(BN_dim, BN_dim, kernel, dilation=1, groups=BN_dim, padding=1)
        # self.endpoint_rnn = nn.LSTM(input_size=BN_dim, hidden_size=BN_dim, num_layers=2, dropout=0, bidirectional=False)
        # self.endpoint_rnn = nn.LSTM(input_size=BN_dim, hidden_size=int(BN_dim/2), num_layers=2, dropout=0, bidirectional=True)
        self.endpoint_conv1d = nn.Conv1d(BN_dim, 1, 1)

    def forward(self, input, voiceP=None, return_features=False):
        # input shape: (B, N, L)
        # normalization
        output = self.BN(self.LN(input))
        # pass to TCN
        # layer 8 stack 3
        # endpoints = []
        features = []

        def append_output(feature_lst, output, return_features=False):
            if return_features:
                feature_lst.append(output)
        if self.skip:
            skip_connection = 0.
            for i in range(len(self.TCN)):
                if i == 0:
                    output, endpoint_0 = self.modulation(
                        output, self.proj[0](voiceP))
                elif i == self.layer: 
                    output, endpoint_1 = self.modulation(
                        output, self.proj[1](voiceP))
                elif i == (self.layer * 2):
                    output, endpoint_2 = self.modulation(
                        output, self.proj[2](voiceP))
                residual, skip = self.TCN[i](output)
                output = output + residual
                if i == (self.layer - 1) or i == (self.layer * 2 - 1)  or i == (self.layer * 3 - 1):
                    append_output(features, output, return_features=return_features)
                skip_connection = skip_connection + skip
        else:
            for i in range(len(self.TCN)):
                residual = self.TCN[i](output)
                output = output + residual
        skip_connection, endpoint_3 = self.modulation(
            skip_connection, self.proj[3](voiceP))
        # skip_connection, endpoint = self.modulation(skip_connection, self.proj[-1](voiceP))
        # endpoints.append(endpoint)

        print('Cues == voiceprint_onset_offset is', self.cues == 'voiceprint_onset_offset')
        # # why???
        # if self.cues == 'voiceprint_onset_offset':
        #     skip_connection = skip_connection / self.stack / \
        #         self.layer  # comment it when no voiceprint cue or onset + voiceprint cues
        # output layer
        if self.skip:
            # output = self.output(skip_connection)
            output = self.output_act(skip_connection)
        else:
            # output = self.output(output)
            output = self.output_act(output)
        output = self.output_conv(output)
        # return output, endpoint_0, endpoint_1, endpoint_2, endpoint_3
        if not return_features:
            # return output, endpoints
            return output, endpoint_0, endpoint_1, endpoint_2, endpoint_3
        else:
            # return output, endpoints, features
            return output, endpoint_0, endpoint_1, endpoint_2, endpoint_3, features

    def modulation(self, features, voiceprint):
        if self.cues == "voiceprint":
            features = features * voiceprint
            return features, None
        elif self.cues == "onset_offset":
            onset_offset = self.get_endpoint(features * voiceprint)
            features = features * onset_offset
            return features, onset_offset
        elif self.cues == "voiceprint_onset_offset":
            features = features * voiceprint
            onset_offset = self.get_endpoint(features)
            features = features * onset_offset
            return features, onset_offset

    def get_endpoint(self, output, layer="none", feature_binning=True):
        if layer == "none":
            endpoint = torch.sigmoid(self.endpoint_conv1d(output))
        elif layer == "dconv1d":
            endpoint = torch.sigmoid(
                self.endpoint_conv1d(self.endpoint_dconv1d(output)))
        elif layer == "rnn":
            if feature_binning:
                output, rest = self.pad_signal(output, 4, 4)
                output_shape = output.shape
                output = torch.reshape(
                    output, (output_shape[0], output_shape[1], -1, 4))
                output = torch.mean(output, -1)
                endpoint_rnn_output, (h, c) = self.endpoint_rnn(
                    output.permute(2, 0, 1))  # bs * d * steps -> steps * bs * d
                endpoint_rnn_output = endpoint_rnn_output.permute(
                    1, 2, 0)  # steps * bs * d -> bs * d * steps
                endpoint = torch.sigmoid(
                    self.endpoint_conv1d(endpoint_rnn_output))
                endpoint_shape = endpoint.shape
                endpoint = endpoint.unsqueeze(-1).repeat(1, 1, 1, 4).view(
                    endpoint_shape[0], endpoint_shape[1], -1)  # pytorch 1.0.1 没有repeat_interleave
                if rest > 0:
                    endpoint = endpoint[:, :, :-rest]
            else:
                endpoint_rnn_output, (h, c) = self.endpoint_rnn(
                    output.permute(2, 0, 1))  # bs * d * steps -> steps * bs * d
                endpoint_rnn_output = endpoint_rnn_output.permute(
                    1, 2, 0)  # steps * bs * d -> bs * d * steps
                endpoint = torch.sigmoid(
                    self.endpoint_conv1d(endpoint_rnn_output))
        return endpoint

    def pad_signal(self, input, win, stride):

        # input is the waveforms: (B, T) or (B, 1, T)
        # reshape and padding
        if input.dim() not in [2, 3]:
            raise RuntimeError("Input can only be 2 or 3 dimensional.")
        if input.dim() == 2:
            input = input.unsqueeze(1)
        batch_size, nfeat, nsample = input.shape
        rest = win - (stride + nsample % win) % win
        if rest > 0:
            pad = Variable(torch.zeros(batch_size, nfeat, rest)
                           ).type(input.type())
            input = torch.cat([input, pad], 2)
        # pad_aux = Variable(torch.zeros(batch_size, 1, stride)).type(input.type())
        # input = torch.cat([pad_aux, input, pad_aux], 2)
        return input, rest
