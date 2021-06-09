# coding=utf8
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import models
from .min_max_quantization import *

class DepthConv1d_Q(nn.Module):
    def __init__(self, input_channel, hidden_channel, kernel, padding, dilation=1, skip=True, causal=False,
                QA_flag=True, ak=8): 
        super(DepthConv1d_Q, self).__init__()
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
        
        self.QA_flag = QA_flag
        self.ak = ak

    def forward(self, input): 
        if self.QA_flag:
            output = min_max_quantize(input, self.ak)
        output = self.reg1(self.nonlinearity1(self.conv1d(output)))

        if self.QA_flag:
            output = min_max_quantize(output, self.ak)
        if self.causal:
            output = self.reg2(self.nonlinearity2(
                self.dconv1d(output)[:, :, :-self.padding]))
        else:
            output = self.reg2(self.nonlinearity2(self.dconv1d(output)))

        if self.QA_flag:
            output = min_max_quantize(output, self.ak)
        residual = self.res_out(output)

        if self.skip:
            skip = self.skip_out(output)
            return residual, skip
        else:
            return residual

class TCN_Q(nn.Module):
    def __init__(self, input_dim, output_dim, BN_dim, hidden_dim,
                 layer, stack, kernel=3, skip=True, win=16, stride=8,
                 causal=False, dilated=True, low_latency=5, cues="voiceprint_onset_offset", 
                 QA_flag=True, ak=8):
        super(TCN_Q, self).__init__()

        self.QA_flag = QA_flag
        self.ak = ak
        # input is a sequence of features of shape (B, N, L)
        self.layer = layer
        self.stack = stack
        self.win = win
        self.stride = stride
        self.cues = cues
        print('Ques used in the model in TCN_Q is', self.cues)

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
                    self.TCN.append(DepthConv1d_Q(
                        BN_dim, hidden_dim, kernel, dilation=2**i, padding=2**i, skip=skip, causal=causal,
                        QA_flag=self.QA_flag, ak=self.ak))
                else:
                    self.TCN.append(DepthConv1d_Q(
                        BN_dim, hidden_dim, kernel, dilation=1, padding=1, skip=skip, causal=causal,
                        QA_flag=self.QA_flag, ak=self.ak))
                if i == 0 and s == 0:
                    self.receptive_field += kernel
                else:
                    if self.dilated:
                        self.receptive_field += (kernel - 1) * 2**i
                    else:
                        self.receptive_field += (kernel - 1)
        print("Receptive field: {:3d} frames.".format(self.receptive_field))

        self.output_act = nn.PReLU()
        self.output_conv = nn.Conv1d(BN_dim, output_dim, 1)
        self.skip = skip
        self.layer = layer
        self.proj = nn.ModuleList([])
        for s in range(stack + 1):
            self.proj.append(nn.Conv1d(BN_dim, BN_dim, 1))

        self.endpoint_conv1d = nn.Conv1d(BN_dim, 1, 1)

    def forward(self, input, voiceP=None, return_features=False):        
        output = self.LN(input)
        if self.QA_flag: # quantize activation for convolution input!
            output = min_max_quantize(output, self.ak)
        output = self.BN(output)
    
        features = []
 
        if self.QA_flag:
            voiceP = min_max_quantize(voiceP, self.ak)

        def append_output(feature_lst, output, return_features=False):
            if return_features:
                feature_lst.append(output)

        endpoint_0 , endpoint_1, endpoint_2, endpoint_3 = None, None, None, None 
        # pass to TCN
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

        if self.skip:
            output = self.output_act(skip_connection)
        else:
            output = self.output_act(output)
        if self.QA_flag:
            output = min_max_quantize(output, self.ak)
        output = self.output_conv(output)

        if not return_features:
            return output, endpoint_0, endpoint_1, endpoint_2, endpoint_3
        else:
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
        if self.QA_flag:
            output = min_max_quantize(output, self.ak)
        if layer == "none":
            endpoint = torch.sigmoid(self.endpoint_conv1d(output)) # 整数，有可能负值，可以的
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
        return input, rest

# Conv-TasNet
class TasNet_Q(nn.Module):
    def __init__(self, enc_dim=512, feature_dim=128, sr=8000, win=2, stride=1,
                 layer=8, stack=3, kernel=3, num_spk=1, causal=False, low_latency=100,
                 cues='voiceprint_onset_offset', QA_flag=True, ak=8):
        super(TasNet_Q, self).__init__()
        # quantization
        self.QA_flag = QA_flag
        self.ak = ak

        # hyper parameters
        self.num_spk = num_spk

        self.enc_dim = enc_dim
        self.feature_dim = feature_dim

        self.win = win
        self.stride = stride

        self.layer = layer
        self.stack = stack
        self.kernel = kernel

        self.causal = causal
        self.low_latency = low_latency

        self.cues = cues
        
        # TCN separator
        self.TCN = TCN_Q(input_dim=self.enc_dim, output_dim=self.enc_dim*self.num_spk, BN_dim=self.feature_dim,
                           hidden_dim=self.feature_dim*4, layer=self.layer, stack=self.stack, kernel=self.kernel,
                           skip=True, win=self.win, stride=self.stride, causal=self.causal, dilated=True,
                           low_latency=self.low_latency, cues=self.cues,
                           QA_flag=self.QA_flag, ak=self.ak)

        self.receptive_field = self.TCN.receptive_field

        # output decoder
        self.decoder = nn.ConvTranspose1d(
            self.enc_dim, 1, self.win, bias=False, stride=self.stride)

    def pad_signal(self, input):
        # input is the waveforms: (B, T) or (B, 1, T)
        # reshape and padding
        if input.dim() not in [2, 3]:
            raise RuntimeError("Input can only be 2 or 3 dimensional.")

        if input.dim() == 2:
            input = input.unsqueeze(1)
        batch_size = input.size(0)
        nsample = input.size(2)
        rest = self.win - (self.stride + nsample % self.win) % self.win
        if rest > 0:
            pad = Variable(torch.zeros(batch_size, 1, rest)).type(input.type())
            input = torch.cat([input, pad], 2)

        pad_aux = Variable(torch.zeros(
            batch_size, 1, self.stride)).type(input.type())
        input = torch.cat([pad_aux, input, pad_aux], 2)

        return input, rest

    def forward(self, enc_output, voiceP=None, return_features=False):
        batch_size = enc_output.size(0)
        # generate masks
        if not return_features:
            TCN_output, endpoint_0, endpoint_1, endpoint_2, endpoint_3 = self.TCN(
                enc_output, voiceP, return_features=return_features)
        else:
            TCN_output, endpoint_0, endpoint_1, endpoint_2, endpoint_3, features = self.TCN(
                enc_output, voiceP, return_features=return_features)
        masks = torch.sigmoid(TCN_output).view(
            batch_size, self.num_spk, self.enc_dim, -1)  # B, C, N, L

        masked_output_1 = enc_output.unsqueeze(1) * masks  # B, C, N, L
        masked_output_2 = enc_output.unsqueeze(1) * (1 - masks)  # B, C, N, L
        masked_output = torch.cat((masked_output_1, masked_output_2), 1)

        # waveform decoder
        output = self.decoder(masked_output.view(
            batch_size * 2, self.enc_dim, -1))  # B*C, 1, L
        output = output[:, :, self.stride:-
                        self.stride].contiguous()  # B*C, 1, L
        output = output.view(batch_size, 2, -1)  # B, C, T

        if not return_features:
            return output, endpoint_0, endpoint_1, endpoint_2, endpoint_3
        else:
            return output, endpoint_0, endpoint_1, endpoint_2, endpoint_3, features


class wase_q(nn.Module):

    def __init__(self, config, fre_size, frame_num, num_labels, use_cuda, score_fc,
                    QA_flag=True, ak=8, init_linear_bias=0):
        super(wase_q, self).__init__()
        # quantization
        self.QA_flag = QA_flag
        self.ak = ak

        self.config = config
        self.fre_size = fre_size
        self.freeze_voiceprint = self.config.FREEZE_VOICEPRINT
        self.ref_time_domain_coding = True
        self.ref_enc_dim = 128
        self.ref_win = 256  # 32 ms
        self.ref_stride = 64  # 8 ms
        self.ref_encoder = nn.Conv1d(
            1, self.ref_enc_dim, self.ref_win, bias=False, stride=self.ref_stride)
        self.voiceprint_encoder = models.rnn_encoder(input_size=self.ref_enc_dim, hidden_size=256, num_layers=2,
                                                     dropout=0.5, bidirectional=True)
        self.set_metric_fc(score_fc, num_labels)
        if (config.ONSET or config.OFFSET) and config.VOICEPRINT:
            self.cues = "voiceprint_onset_offset"
        elif config.ONSET or config.OFFSET:
            self.cues = "onset_offset"
        elif config.VOICEPRINT:
            self.cues = "voiceprint"
        else:
            raise ValueError(
                "Cues should be in ['voiceprint_onset_offset', 'onset_offset', 'voiceprint']")
        print('Ques used in the model is', self.cues)
        if self.config.BLOCKS == 'tcn':
            self.enc_dim = 512
            self.feature_dim = 128
            self.sr = self.config['FRAME_RATE']
            self.win = 16
            self.layer = config.layer
            self.stack = 3
            self.stride = self.win // 2
            self.causal = self.config['REAL_TIME']
            self.low_latency = self.config['LOW_LATENCY']
            self.linear = nn.Linear(512, self.feature_dim) # voiceprint_linear, onset_offset
            self.encoder = nn.Conv1d(
                1, self.enc_dim, self.win, bias=False, stride=self.stride) # encode mixture
            self.ss_model = TasNet_Q(enc_dim=self.enc_dim, feature_dim=self.feature_dim, sr=self.sr, win=self.win,
                                          stride=self.stride, layer=self.layer, stack=self.stack, kernel=3, num_spk=1,
                                          causal=self.causal, low_latency=self.low_latency, cues=self.cues,
                                          QA_flag=self.QA_flag, ak=self.ak)

        self.criterion = models.criterion(
            num_labels, use_cuda, self.config.loss)

        self.bce_loss = nn.BCELoss()

    def forward(self, mix_wav, ref_wav, ref_wav_len, oracle_wav_endpoint, return_features=False):
        ref_feas, ref_feas_len = self.encode_reference(ref_wav, ref_wav_len)
        voiceprint = self.get_voiceprint(ref_feas, ref_feas_len)
        if self.freeze_voiceprint:
            voiceprint.detach_()  # in place func
        enc_output, rest, oracle_feat_endpoint = self.encode_noisy(
            mix_wav, oracle_wav_endpoint)
        if self.config.BLOCKS == 'tcn':
            if not return_features:
                predicted, endpoint_0, endpoint_1, endpoint_2, endpoint_3 = self.ss_model(
                    enc_output, voiceprint.unsqueeze(-1), return_features)
            else:
                predicted, endpoint_0, endpoint_1, endpoint_2, endpoint_3, features = self.ss_model(
                    enc_output, voiceprint.unsqueeze(-1), return_features)
            if rest > 0:
                predicted = predicted[:, :, :-rest]
            if not return_features:
                return voiceprint, predicted, oracle_feat_endpoint, endpoint_0, endpoint_1, endpoint_2, endpoint_3
            elif return_features:
                return voiceprint, predicted, oracle_feat_endpoint, endpoint_0, endpoint_1, endpoint_2, endpoint_3, features

        elif self.config.BLOCKS == 'dprnn':
            predicted, endpoint_list_tensor, endpoint_last, oracle_chunk_endpoint = self.ss_model(
                enc_output, voiceprint.unsqueeze(-1), oracle_feat_endpoint)
            if rest > 0:
                predicted = predicted[:, :, :-rest]
            return voiceprint, predicted, oracle_chunk_endpoint, oracle_feat_endpoint, endpoint_list_tensor, endpoint_last

    def test(self, mix_wav, ref_wav, ref_wav_len, oracle_wav_endpoint, return_features=False):
        ref_feas, ref_feas_len = self.encode_reference(ref_wav, ref_wav_len)
        voiceprint = self.get_voiceprint(ref_feas, ref_feas_len)
        enc_output, rest, oracle_feat_endpoint = self.encode_noisy(
            mix_wav, oracle_wav_endpoint)
        if self.config.BLOCKS == 'tcn':
            if not return_features:
                predicted, endpoint_0, endpoint_1, endpoint_2, endpoint_3 = self.ss_model(
                    enc_output, voiceprint.unsqueeze(-1), return_features)
            else:
                predicted, endpoint_0, endpoint_1, endpoint_2, endpoint_3, features = self.ss_model(
                    enc_output, voiceprint.unsqueeze(-1), return_features)
            if rest > 0:
                predicted = predicted[:, :, :-rest]
            if not return_features:
                return predicted, oracle_feat_endpoint, endpoint_0, endpoint_1, endpoint_2, endpoint_3
            else:
                return predicted, oracle_feat_endpoint, endpoint_0, endpoint_1, endpoint_2, endpoint_3, features
        elif self.config.BLOCKS == 'dprnn':
            predicted, endpoint_list_tensor, endpoint_last, oracle_chunk_endpoint = self.ss_model(
                enc_output, voiceprint.unsqueeze(-1), oracle_feat_endpoint)
            if rest > 0:
                predicted = predicted[:, :, :-rest]
            return predicted, oracle_chunk_endpoint, oracle_feat_endpoint, endpoint_list_tensor, endpoint_last

    def encode_reference(self, ref_wav, ref_wav_len, ref_stft_feas=None, ref_stft_feas_len=None):
        if self.ref_time_domain_coding:
            ref_output, rest = self.pad_signal(
                ref_wav, self.ref_win, self.ref_stride)
            ref_enc_output = self.ref_encoder(
                ref_output).transpose(1, 2)  # B, N, L -> B, L, N
            ref_feas_len = (ref_wav_len + rest - self.ref_win) / \
                self.ref_stride + 1
            return ref_enc_output, ref_feas_len
        else:
            return ref_stft_feas, ref_stft_feas_len

    def get_voiceprint(self, ref_feas, ref_feas_len, pooling_method='mean'):
        ref_feas_len_sorted, indices = torch.sort(
            ref_feas_len, dim=0, descending=True)
        _, indices_unsort = torch.sort(indices, dim=0, descending=False)
        ref_feas_sorted = torch.index_select(
            ref_feas, dim=0, index=indices)  # bs * steps * d
        voiceprint, state = self.voiceprint_encoder(
            ref_feas_sorted, ref_feas_len_sorted.tolist())  # steps * bs * (d * dirction)
        if pooling_method == 'mean':
            # bs * steps * d -> bs * d
            voiceprint = torch.sum(voiceprint, dim=1) / \
                ref_feas_len_sorted.unsqueeze(-1)
            voiceprint = torch.index_select(
                voiceprint, dim=0, index=indices_unsort)  # bs * d
        elif pooling_method == 'max':
            voiceprint, _ = torch.max(voiceprint, dim=0)
        voiceprint = self.linear(voiceprint)
        return voiceprint

    def encode_noisy(self, input, oracle_wav_endpoint=None):
        output, rest = self.pad_signal(input, self.win, self.stride)
        if self.config.BLOCKS == 'tcn':
            enc_output = self.encoder(output)  # B, N, L
        elif self.config.BLOCKS == 'dprnn':
            enc_output = F.relu(self.encoder(output))  # B, E, L
        if oracle_wav_endpoint is not None:
            oracle_feat_endpoint, _ = self.pad_signal(
                oracle_wav_endpoint, self.win, self.stride)
            oracle_feat_endpoint = oracle_feat_endpoint[:, :, :-self.stride]
            # B, T -> B, N
            oracle_feat_endpoint = oracle_feat_endpoint[:, :, ::self.stride]
        return enc_output, rest, oracle_feat_endpoint

    def pad_signal(self, input, win, stride):
        # input is the waveforms: (B, T) or (B, 1, T)
        # reshape and padding
        if input.dim() not in [2, 3]:
            raise RuntimeError("Input can only be 2 or 3 dimensional.")

        if input.dim() == 2:
            input = input.unsqueeze(1)
        batch_size = input.size(0)
        nsample = input.size(2)
        rest = win - (stride + nsample % win) % win
        if rest > 0:
            pad = Variable(torch.zeros(batch_size, 1, rest)).type(input.type())
            input = torch.cat([input, pad], 2)

        pad_aux = Variable(torch.zeros(
            batch_size, 1, stride)).type(input.type())
        input = torch.cat([pad_aux, input, pad_aux], 2)
        return input, rest

    def set_metric_fc(self, score_fc, num_labels):
        if self.config.BLOCKS == 'tcn':
            self.config.VOICEP_EMB_SIZE = 128
        elif self.config.BLOCKS == 'dprnn':
            self.config.VOICEP_EMB_SIZE = 64
        self.score_fc = score_fc
        if score_fc == 'add_margin':
            self.metric_fc = models.AddMarginProduct(
                self.config.VOICEP_EMB_SIZE, num_labels, s=30, m=0.35)
        elif score_fc == 'arc_margin':
            self.metric_fc = models.ArcMarginProduct(
                self.config.VOICEP_EMB_SIZE, num_labels, s=30, m=0.5, easy_margin=False)
        elif score_fc == 'sphere':
            self.metric_fc = models.SphereProduct(
                self.config.VOICEP_EMB_SIZE, num_labels, m=4)
        elif score_fc == 'linear':
            self.metric_fc = nn.Linear(self.config.VOICEP_EMB_SIZE, num_labels)
        else:
            raise ValueError('opt.score_fc')

    def voiceprint_loss(self, voiceprint, targets):
        return models.cross_entropy_loss(voiceprint, targets, self.criterion, self.metric_fc, self.score_fc)

    def separation_tas_loss(self, aim_wav, predicted, mix_lengths):
        return models.ss_tas_loss(aim_wav, predicted, mix_lengths)

    def separation_tas_mse_loss(self, aim_wav, predicted, mix_lengths):
        return models.cal_mse_loss_with_order(aim_wav, predicted, mix_lengths)

    def endpoint_loss(self, endpoint, oracle_endpoint):
        # ACC F1
        if self.cues == "voiceprint":
            endpoint = torch.ones_like(oracle_endpoint)
            zero = torch.tensor(0.0).cuda()
            return self.bce_loss(endpoint, oracle_endpoint), zero, zero, zero, zero
        binary_endpoint = (endpoint > 0.5).type_as(oracle_endpoint)
        acc = torch.mean(torch.eq(binary_endpoint, oracle_endpoint).type_as(
            oracle_endpoint)) / torch.numel(oracle_endpoint)
        TP_point = (binary_endpoint + oracle_endpoint >
                    1).type_as(oracle_endpoint)
        TP = torch.sum(TP_point).type_as(oracle_endpoint)
        TN = torch.sum((binary_endpoint + oracle_endpoint <
                        1).type_as(oracle_endpoint))
        FN = torch.sum((oracle_endpoint - TP_point ==
                        1).type_as(oracle_endpoint))
        FP = torch.sum((binary_endpoint - TP_point ==
                        1).type_as(oracle_endpoint))
        total_num = torch.numel(oracle_endpoint)
        accuracy = (TP + TN) / (total_num + 1e-6)
        precision = TP / (TP + FP + 1e-6)
        recall = TP / (TP + FN + 1e-6)
        f1 = 2 * precision * recall / (precision + recall + 1e-6)
        return self.bce_loss(endpoint, oracle_endpoint), accuracy, precision, recall, f1