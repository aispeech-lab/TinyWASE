# coding=utf8
import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
from itertools import permutations
import librosa

EPS = 1e-8


class CenterLoss(nn.Module):
    """Center loss.

    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.

    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """

    def __init__(self, num_classes=10, feat_dim=2, use_gpu=True):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu

        if self.use_gpu:
            self.centers = nn.Parameter(torch.randn(
                self.num_classes, self.feat_dim).cuda())
        else:
            self.centers = nn.Parameter(
                torch.randn(self.num_classes, self.feat_dim))

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        batch_size = x.size(0)
        labels = torch.tensor(labels).cuda().view(-1)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
            torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(
                self.num_classes, batch_size).t()
        distmat.addmm_(1, -2, x, self.centers.t())

        classes = torch.arange(self.num_classes).long()
        if self.use_gpu:
            classes = classes.cuda()
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = distmat * mask.float()
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size

        return loss, torch.tensor(x.size(0)), torch.tensor(0)


class triplet_loss(nn.Module):
    def __init__(self):
        super(triplet_loss, self).__init__()
        self.w = nn.Parameter(torch.Tensor(1))
        self.b = nn.Parameter(torch.Tensor(1))
        self.reset_parameters()

    def reset_parameters(self):
        self.w.data.fill_(1)
        self.b.data.zero_()

    def forward(self, voiceP_outputs, tuple_voiceP_outputs_center, tuple_label):
        cosine_distance = F.cosine_similarity(
            voiceP_outputs, tuple_voiceP_outputs_center, dim=-1)
        candidate_loss = F.sigmoid(self.w * cosine_distance + self.b)
        sign = (torch.zeros_like(tuple_label) == tuple_label).type_as(
            tuple_label) - tuple_label
        pred = torch.gt(candidate_loss, 0.5).type(
            torch.FloatTensor) - torch.lt(candidate_loss, 0.5).type(torch.FloatTensor)
        pred = (pred + 1) / 2
        num_correct = pred.data.eq(tuple_label.type(torch.FloatTensor)).sum()
        loss = torch.mean(sign.type_as(candidate_loss) * candidate_loss, dim=0)
        loss2 = torch.mean(tuple_label.float().cuda(
        )*candidate_loss + (1-tuple_label.float().cuda())*(-candidate_loss))
        print('loss: %6.3f, loss2: %6.3f, num_sample: %d, num_correct: %d' %
              (loss, loss2, len(tuple_label), num_correct))
        return loss, torch.tensor(len(tuple_label)), num_correct


# reference https://github.com/jonlu0602/DeepDenoisingAutoencoder/blob/master/python/utils.py
class WaveLoss(nn.Module):
    def __init__(self, dBscale=1, denormalize=1, max_db=100, ref_db=20, nfft=512, hop_size=int(0.004 * 16000)):  # 参数这样放着比较危险
        super(WaveLoss, self).__init__()
        self.dBscale = dBscale
        self.denormalize = denormalize
        self.max_db = max_db
        self.ref_db = ref_db
        self.nfft = nfft
        self.hop_size = hop_size
        self.mse_loss = nn.MSELoss()

    def genWav(self, real, imag):
        '''
        :param S: (B, F-1, T) to be padded with 0 in this function
        :param phase: (B, F, T)
        :return: (B, num_samples)
        '''
        predicted_complex = real.data.cpu().numpy() + (1j * imag.data.cpu().numpy())
        wav_pre = librosa.core.spectrum.istft(
            np.transpose(predicted_complex), self.hop_size)
        return wav_pre

        # stft_matrix = torch.stack([real, imag], dim=4).transpose(3,2)  #(B,top_k,F,T,2)
        # wav1 = istft_irfft(stft_matrix[:,0,:,:,:], length=48000, hop_length=self.hop_size, win_length=self.nfft)
        # wav2 = istft_irfft(stft_matrix[:,1,:,:,:], length=48000, hop_length=self.hop_size, win_length=self.nfft)
        # return wav1, wav2

    def forward(self, target_real, target_imag, pred_real, pred_imag):
        '''
        :param target_mag: (B, F-1, T)
        :param target_phase: (B, F, T)
        :param pred_mag: (B, F-1, T)
        :param pred_phase: (B, F, T)
        :return:
        '''
        # target_wav_1, target_wav_2 = self.genWav(target_real, target_imag)
        # pred_wav_1, pred_wav_2 = self.genWav(pred_real, pred_imag)

        target_wav_1 = self.genWav(target_real, target_imag)
        pred_wav_1 = self.genWav(pred_real, pred_imag)

        # target_wav_arr = target_wav.squeeze(0).cpu().data.numpy()
        # pred_wav_arr = pred_wav.squeeze(0).cpu().data.numpy()
        # print('target wav arr', target_wav_arr.shape)
        # sf.write('target.wav', target_wav_arr, 8000)
        # sf.write('pred.wav', pred_wav_arr, 8000)

        # loss = 100 * (self.mse_loss(target_wav_1, pred_wav_1) + self.mse_loss(target_wav_2, pred_wav_2)) / 2
        loss = 100 * self.mse_loss(target_wav_1, pred_wav_1) / 2
        return loss


class FocalLoss(nn.Module):

    def __init__(self, gamma=0, eps=1e-7):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps
        self.ce = torch.nn.CrossEntropyLoss()

    def forward(self, input, target):
        logp = self.ce(input, target)
        p = torch.exp(-logp)
        loss = (1 - p) ** self.gamma * logp
        return loss.mean()


def rank_feas(raw_tgt, feas_list, out_type='torch'):
    final_num = []
    for each_feas, each_line in zip(feas_list, raw_tgt):
        for spk in each_line:
            final_num.append(each_feas[spk])
    if out_type == 'numpy':
        return np.array(final_num)
    else:
        return torch.from_numpy(np.array(final_num))


def criterion(tgt_vocab_size, use_cuda, loss):
    weight = torch.ones(tgt_vocab_size)
    if loss == 'focal_loss':
        crit = FocalLoss(gamma=2)
    else:
        crit = nn.CrossEntropyLoss(weight, size_average=False)
    if use_cuda:
        crit.cuda()
    return crit


def compute_score(hiddens, targets, metric_fc, score_fc='arc_margin'):
    if score_fc in ['add_margin', 'arc_margin', 'sphere']:
        scores = metric_fc(hiddens, targets)
    elif score_fc == 'linear':
        scores = metric_fc(hiddens)
    else:
        raise ValueError(
            "score_fc should be in ['add_margin', 'arc_margin', 'sphere' and 'linear']")
    return scores


def cross_entropy_loss(hidden_outputs, targets, criterion, metric_fc, score_fc):
    targets = torch.tensor(targets).cuda().view(-1)
    scores = compute_score(hidden_outputs, targets, metric_fc, score_fc)
    loss = criterion(scores, targets)
    pred = scores.max(1)[1]
    num_correct = pred.eq(targets).sum()
    num_total = targets.size()[0]
    loss = loss.div(num_total)
    return loss, torch.tensor(num_total).cuda(), torch.tensor(num_correct).cuda()


def ss_loss(config, x_input_map_multi, multi_mask, y_multi_map, loss_multi_func, wav_loss):
    predict_multi_map = multi_mask * x_input_map_multi
    y_multi_map = Variable(y_multi_map)
    loss_multi_speech = loss_multi_func(predict_multi_map, y_multi_map)
    return loss_multi_speech


def ss_tas_loss(aim_wav, predicted, mix_length):
    # 原来是返回 [0]的？？
    loss = cal_loss_with_order(aim_wav, predicted, mix_length)
    return loss


def cal_loss_with_order(source, estimate_source, source_lengths):
    """
    Args:
        source: [B, C, T], B is batch size
        estimate_source: [B, C, T]
        source_lengths: [B]
    """
    max_snr = cal_si_snr_with_order(source, estimate_source, source_lengths)
    loss = 0 - torch.mean(max_snr)
    return loss


def cal_loss_with_PIT(source, estimate_source, source_lengths):
    """
    Args:
        source: [B, C, T], B is batch size
        estimate_source: [B, C, T]
        source_lengths: [B]
    """
    max_snr, perms, max_snr_idx = cal_si_snr_with_pit(source,
                                                      estimate_source,
                                                      source_lengths)
    loss = 0 - torch.mean(max_snr)
    reorder_estimate_source = reorder_source(
        estimate_source, perms, max_snr_idx)
    return loss, max_snr, estimate_source, reorder_estimate_source


def cal_si_snr_with_order(source, estimate_source, source_lengths):
    """Calculate SI-SNR with given order.
    Args:
        source: [B, C, T], B is batch size
        estimate_source: [B, C, T]
        source_lengths: [B], each item is between [0, T]
    """
    assert source.size() == estimate_source.size()
    B, C, T = source.size()
    # mask padding position along T
    mask = get_mask(source, source_lengths)
    estimate_source *= mask

    # Step 1. Zero-mean norm
    num_samples = source_lengths.view(-1, 1, 1).float()  # [B, 1, 1]
    mean_target = torch.sum(source, dim=2, keepdim=True) / num_samples
    mean_estimate = torch.sum(estimate_source, dim=2,
                              keepdim=True) / num_samples
    zero_mean_target = source - mean_target
    zero_mean_estimate = estimate_source - mean_estimate
    # mask padding position along T
    zero_mean_target *= mask
    zero_mean_estimate *= mask

    # Step 2. SI-SNR with order
    # reshape to use broadcast
    s_target = zero_mean_target  # [B, C, T]
    s_estimate = zero_mean_estimate  # [B, C, T]
    # s_target = <s', s>s / ||s||^2
    pair_wise_dot = torch.sum(s_estimate * s_target,
                              dim=2, keepdim=True)  # [B, C, 1]
    s_target_energy = torch.sum(
        s_target ** 2, dim=2, keepdim=True) + EPS  # [B, C, 1]
    pair_wise_proj = pair_wise_dot * s_target / s_target_energy  # [B, C, T]
    # e_noise = s' - s_target
    e_noise = s_estimate - pair_wise_proj  # [B, C, T]
    # SI-SNR = 10 * log_10(||s_target||^2 / ||e_noise||^2)
    pair_wise_si_snr = torch.sum(
        pair_wise_proj ** 2, dim=2) / (torch.sum(e_noise ** 2, dim=2) + EPS)
    pair_wise_si_snr = 10 * torch.log10(pair_wise_si_snr + EPS)  # [B, C]
    print(pair_wise_si_snr)

    return torch.sum(pair_wise_si_snr, dim=1)/C

def cal_mse_loss_with_order(source, estimate_source, source_lengths):
    """Calculate SI-SNR with given order.
    Args:
        source: [B, C, T], B is batch size
        estimate_source: [B, C, T]
        source_lengths: [B], each item is between [0, T]
    """
    assert source.size() == estimate_source.size()
    B, C, T = source.size()
    # mask padding position along T
    mask = get_mask(source, source_lengths)
    estimate_source *= mask

    # Step 1. Zero-mean norm
    num_samples = source_lengths.view(-1, 1, 1).float()  # [B, 1, 1]
    mean_target = torch.sum(source, dim=2, keepdim=True) / num_samples
    mean_estimate = torch.sum(estimate_source, dim=2,
                              keepdim=True) / num_samples
    zero_mean_target = source - mean_target
    zero_mean_estimate = estimate_source - mean_estimate
    # mask padding position along T
    zero_mean_target *= mask
    zero_mean_estimate *= mask

    # Step 2. SI-SNR with order
    # reshape to use broadcast
    s_target = zero_mean_target  # [B, C, T]
    s_estimate = zero_mean_estimate  # [B, C, T]
    mse_loss = nn.MSELoss()
    loss = mse_loss(s_target, s_estimate)
    return loss


def cal_si_snr_with_pit(source, estimate_source, source_lengths):
    """Calculate SI-SNR with PIT training.
    Args:
        source: [B, C, T], B is batch size
        estimate_source: [B, C, T]
        source_lengths: [B], each item is between [0, T]
    """
    assert source.size() == estimate_source.size()
    B, C, T = source.size()
    # mask padding position along T
    mask = get_mask(source, source_lengths)
    estimate_source *= mask

    # Step 1. Zero-mean norm
    num_samples = source_lengths.view(-1, 1, 1).float()  # [B, 1, 1]
    mean_target = torch.sum(source, dim=2, keepdim=True) / num_samples
    mean_estimate = torch.sum(estimate_source, dim=2,
                              keepdim=True) / num_samples
    zero_mean_target = source - mean_target
    zero_mean_estimate = estimate_source - mean_estimate
    # mask padding position along T
    zero_mean_target *= mask
    zero_mean_estimate *= mask

    # Step 2. SI-SNR with PIT
    # reshape to use broadcast
    s_target = torch.unsqueeze(zero_mean_target, dim=1)  # [B, 1, C, T]
    s_estimate = torch.unsqueeze(zero_mean_estimate, dim=2)  # [B, C, 1, T]
    # s_target = <s', s>s / ||s||^2
    pair_wise_dot = torch.sum(s_estimate * s_target,
                              dim=3, keepdim=True)  # [B, C, C, 1]
    s_target_energy = torch.sum(
        s_target ** 2, dim=3, keepdim=True) + EPS  # [B, 1, C, 1]
    pair_wise_proj = pair_wise_dot * s_target / s_target_energy  # [B, C, C, T]
    # e_noise = s' - s_target
    e_noise = s_estimate - pair_wise_proj  # [B, C, C, T]
    # SI-SNR = 10 * log_10(||s_target||^2 / ||e_noise||^2)
    pair_wise_si_snr = torch.sum(
        pair_wise_proj ** 2, dim=3) / (torch.sum(e_noise ** 2, dim=3) + EPS)
    pair_wise_si_snr = 10 * torch.log10(pair_wise_si_snr + EPS)  # [B, C, C]

    # Get max_snr of each utterance
    # permutations, [C!, C]
    perms = source.new_tensor(list(permutations(range(C))), dtype=torch.long)
    # one-hot, [C!, C, C]
    index = torch.unsqueeze(perms, 2)
    # perms_one_hot = source.new_zeros((*perms.size(), C)).scatter_(2, index, 1)
    perms_one_hot = source.new_zeros(
        (perms.size()[0], perms.size()[1], C)).scatter_(2, index, 1)
    # [B, C!] <- [B, C, C] einsum [C!, C, C], SI-SNR sum of each permutation
    snr_set = torch.einsum('bij,pij->bp', [pair_wise_si_snr, perms_one_hot])
    max_snr_idx = torch.argmax(snr_set, dim=1)  # [B]
    # max_snr = torch.gather(snr_set, 1, max_snr_idx.view(-1, 1))  # [B, 1]
    max_snr, _ = torch.max(snr_set, dim=1, keepdim=True)
    max_snr /= C
    return max_snr, perms, max_snr_idx


def reorder_source(source, perms, max_snr_idx):
    """
    Args:
        source: [B, C, T]
        perms: [C!, C], permutations
        max_snr_idx: [B], each item is between [0, C!)
    Returns:
        reorder_source: [B, C, T]
    """
    # B, C, *_ = source.size()
    B, C, __ = source.size()
    # [B, C], permutation whose SI-SNR is max of each utterance
    # for each utterance, reorder estimate source according this permutation
    max_snr_perm = torch.index_select(perms, dim=0, index=max_snr_idx)
    # print('max_snr_perm', max_snr_perm)
    # maybe use torch.gather()/index_select()/scatter() to impl this?
    reorder_source = torch.zeros_like(source)
    for b in range(B):
        for c in range(C):
            reorder_source[b, c] = source[b, max_snr_perm[b][c]]
    return reorder_source


def get_mask(source, source_lengths):
    """
    Args:
        source: [B, C, T]
        source_lengths: [B]
    Returns:
        mask: [B, 1, T]
    """
    B, _, T = source.size()
    mask = source.new_ones((B, 1, T))
    for i in range(B):
        mask[i, :, source_lengths[i]:] = 0
    return mask


def ss_loss_MLMSE(config, x_input_map_multi, multi_mask, y_multi_map, loss_multi_func, Var):
    try:
        if Var == None:
            Var = Variable(torch.eye(
                config.fre_size, config.fre_size).cuda(), requires_grad=0)
            print('Set Var to:', Var)
    except:
        pass
    assert Var.size() == (config.fre_size, config.fre_size)

    predict_multi_map = torch.mean(
        multi_mask * x_input_map_multi, -2)
    # predict_multi_map=Variable(y_multi_map)
    y_multi_map = torch.mean(Variable(y_multi_map), -2)

    loss_vector = (y_multi_map - predict_multi_map).view(-1,
                                                         config.fre_size).unsqueeze(1)

    Var_inverse = torch.inverse(Var)
    Var_inverse = Var_inverse.unsqueeze(0).expand(loss_vector.size()[0], config.fre_size,
                                                  config.fre_size)
    loss_multi_speech = torch.bmm(
        torch.bmm(loss_vector, Var_inverse), loss_vector.transpose(1, 2))
    loss_multi_speech = torch.mean(loss_multi_speech, 0)

    y_sum_map = Variable(torch.ones(
        config.batch_size, config.frame_num, config.fre_size)).cuda()
    predict_sum_map = torch.sum(multi_mask, 1)
    loss_multi_sum_speech = loss_multi_func(predict_sum_map, y_sum_map)
    print('loss 1 eval, losssum eval : ', loss_multi_speech.data.cpu(
    ).numpy(), loss_multi_sum_speech.data.cpu().numpy())
    # loss_multi_speech=loss_multi_speech+0.5*loss_multi_sum_speech
    print('evaling multi-abs norm this eval batch:',
          torch.abs(y_multi_map - predict_multi_map).norm().data.cpu().numpy())
    # loss_multi_speech=loss_multi_speech+3*loss_multi_sum_speech
    print('loss for whole separation part:',
          loss_multi_speech.data.cpu().numpy())
    # return F.relu(loss_multi_speech)
    return loss_multi_speech


def dis_loss(config, top_k_num, dis_model, x_input_map_multi, multi_mask, y_multi_map, loss_multi_func):
    predict_multi_map = multi_mask * x_input_map_multi
    y_multi_map = Variable(y_multi_map).cuda()
    score_true = dis_model(y_multi_map)
    score_false = dis_model(predict_multi_map)
    acc_true = torch.sum(score_true > 0.5).data.cpu(
    ).numpy() / float(score_true.size()[0])
    acc_false = torch.sum(score_false < 0.5).data.cpu(
    ).numpy() / float(score_true.size()[0])
    acc_dis = (acc_false + acc_true) / 2
    print('acc for dis:(ture,false,aver)', acc_true, acc_false, acc_dis)

    loss_dis_true = loss_multi_func(score_true, Variable(
        torch.ones(config.batch_size * top_k_num, 1)).cuda())
    loss_dis_false = loss_multi_func(score_false, Variable(
        torch.zeros(config.batch_size * top_k_num, 1)).cuda())
    loss_dis = loss_dis_true + loss_dis_false
    print('loss for dis:(ture,false)', loss_dis_true.data.cpu().numpy(),
          loss_dis_false.data.cpu().numpy())
    return loss_dis
