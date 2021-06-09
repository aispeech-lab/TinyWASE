# coding=utf8
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import models


class wase(nn.Module):

    def __init__(self, config, fre_size, frame_num, num_labels, use_cuda, score_fc, teacher=True, sharing=0):
        super(wase, self).__init__()
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
        print('Cues used in the model is', self.cues)
        if self.config.BLOCKS == 'tcn':
            self.enc_dim = 512
            self.feature_dim = 128
            self.sr = self.config['FRAME_RATE']
            self.win = 16
            if teacher:
                self.layer = config.layer_teacher # config.layer # 8
            else:
                self.layer = config.layer_student
            self.stack = config.stack # 3
            self.stride = self.win // 2
            self.causal = self.config['REAL_TIME']
            self.low_latency = self.config['LOW_LATENCY']
            self.linear = nn.Linear(512, self.feature_dim) # voiceprint_linear, onset_offset
            self.encoder = nn.Conv1d(
                1, self.enc_dim, self.win, bias=False, stride=self.stride)
            self.ss_model = models.TasNet(enc_dim=self.enc_dim, feature_dim=self.feature_dim, sr=self.sr, win=self.win,
                                          stride=self.stride, layer=self.layer, stack=self.stack, kernel=3, num_spk=1,
                                          causal=self.causal, low_latency=self.low_latency, cues=self.cues, sharing=sharing)

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
                predicted, endpoint_0, endpoint_1, endpoint_2, endpoint_3 = self.ss_model(enc_output, voiceprint.unsqueeze(-1), return_features=return_features)
            else:
                predicted, endpoint_0, endpoint_1, endpoint_2, endpoint_3, features = self.ss_model(enc_output, voiceprint.unsqueeze(-1), return_features=return_features)                

            if rest > 0:
                predicted = predicted[:, :, :-rest]
            if not return_features:
                return voiceprint, predicted, oracle_feat_endpoint, endpoint_0, endpoint_1, endpoint_2, endpoint_3
            else:
                return voiceprint, predicted, oracle_feat_endpoint, endpoint_0, endpoint_1, endpoint_2, endpoint_3, features


    def test(self, mix_wav, ref_wav, ref_wav_len, oracle_wav_endpoint, return_features=False):
        ref_feas, ref_feas_len = self.encode_reference(ref_wav, ref_wav_len)
        voiceprint = self.get_voiceprint(ref_feas, ref_feas_len)
        enc_output, rest, oracle_feat_endpoint = self.encode_noisy(
            mix_wav, oracle_wav_endpoint)
        if self.config.BLOCKS == 'tcn':
            if not return_features:
                predicted, endpoint_0, endpoint_1, endpoint_2, endpoint_3 = self.ss_model(enc_output, voiceprint.unsqueeze(-1), return_features)
            else:
                predicted, endpoint_0, endpoint_1, endpoint_2, endpoint_3, features = self.ss_model(enc_output, voiceprint.unsqueeze(-1), return_features)
            if rest > 0:
                predicted = predicted[:, :, :-rest]
            if not return_features:
                return predicted, oracle_wav_endpoint, endpoint_0, endpoint_1, endpoint_2, endpoint_3
            else:
                return predcited, oracle_wav_endpoint, endpoint_0, endpoint_1, endpoint_2, endpoint_3, features

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
