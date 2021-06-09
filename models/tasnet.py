import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

# Conv-TasNet
class TasNet(nn.Module):
    def __init__(self, enc_dim=512, feature_dim=128, sr=8000, win=2, stride=1,
                 layer=8, stack=3, kernel=3, num_spk=1, causal=False, low_latency=100,
                 cues='voiceprint_onset_offset', sharing=0):
        super(TasNet, self).__init__()

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
        if sharing:
            import models.tcn_pshare as tcn
        else:
            import models.tcn as tcn
        self.TCN = tcn.TCN(input_dim=self.enc_dim, output_dim=self.enc_dim*self.num_spk, BN_dim=self.feature_dim,
                        hidden_dim=self.feature_dim*4, layer=self.layer, stack=self.stack, kernel=self.kernel,
                        skip=True, win=self.win, stride=self.stride, causal=self.causal, dilated=True,
                        low_latency=self.low_latency, cues=self.cues)

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
        if not return_features:
            TCN_output, endpoint_0, endpoint_1, endpoint_2, endpoint_3 = self.TCN(enc_output, voiceP, return_features=return_features)
        else:
            TCN_output, endpoint_0, endpoint_1, endpoint_2, endpoint_3, features = self.TCN(enc_output, voiceP, return_features=return_features)

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


def test_conv_tasnet():
    x = torch.rand(2, 32000)
    nnet = TasNet()
    x = nnet(x)
    s1 = x[0]
    print(s1.shape)


if __name__ == "__main__":
    test_conv_tasnet()
