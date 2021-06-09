# coding=utf8
import numpy as np
import os
import soundfile as sf
from utils.separation import bss_eval_sources


def cal_SDRi(src_ref, src_est, mix, permutation=False):
    """Calculate Source-to-Distortion Ratio improvement (SDRi).
    NOTE: bss_eval_sources is very very slow.
    Args:
        src_ref: numpy.ndarray, [C, T]
        src_est: numpy.ndarray, [C, T], reordered by best PIT permutation
        mix: numpy.ndarray, [T]
    Returns:
        average_SDRi
    """
    src_anchor = np.stack([mix, mix], axis=0)
    if src_ref.shape[0] == 1:
        src_anchor = src_anchor[0]
    sdr, sir, sar, popt = bss_eval_sources(
        src_ref, src_est, compute_permutation=permutation)
    sdr0, sir0, sar0, popt0 = bss_eval_sources(
        src_ref, src_anchor, compute_permutation=permutation)
    # avg_SDRi = ((sdr[0]-sdr0[0]) + (sdr[1]-sdr0[1])) / 2
    avg_SDRi = sdr[0] - sdr0[0]
    # print("SDRi1: {0:.2f}, SDRi2: {1:.2f}".format(sdr[0]-sdr0[0], sdr[1]-sdr0[1]))
    return avg_SDRi


def cal_SISNRi(src_ref, src_est, mix):
    """Calculate Scale-Invariant Source-to-Noise Ratio improvement (SI-SNRi)
    Args:
        src_ref: numpy.ndarray, [C, T]
        src_est: numpy.ndarray, [C, T], reordered by best PIT permutation
        mix: numpy.ndarray, [T]
    Returns:
        average_SISNRi
    """
    sisnr1 = cal_SISNR(src_ref[0], src_est[0])
    # sisnr2 = cal_SISNR(src_ref[1], src_est[1])
    sisnr1b = cal_SISNR(src_ref[0], mix)
    # sisnr2b = cal_SISNR(src_ref[1], mix)
    # print("SISNR base1 {0:.2f} SISNR base2 {1:.2f}, avg {2:.2f}".format(
    #     sisnr1b, sisnr2b, (sisnr1b+sisnr2b) / 2))
    # print("SISNRi1: {0:.2f}, SISNRi2: {1:.2f}".format(sisnr1, sisnr2))
    # avg_SISNRi = ((sisnr1 - sisnr1b) + (sisnr2 - sisnr2b)) / 2
    avg_SISNRi = sisnr1 - sisnr1b
    return avg_SISNRi


def cal_SISNR(ref_sig, out_sig, eps=1e-8):
    """Calcuate Scale-Invariant Source-to-Noise Ratio (SI-SNR)
    Args:
        ref_sig: numpy.ndarray, [T]
        out_sig: numpy.ndarray, [T]
    Returns:
        SISNR
    """
    assert len(ref_sig) == len(out_sig)
    ref_sig = ref_sig - np.mean(ref_sig)
    out_sig = out_sig - np.mean(out_sig)
    ref_energy = np.sum(ref_sig ** 2) + eps
    proj = np.sum(ref_sig * out_sig) * ref_sig / ref_energy
    noise = out_sig - proj
    ratio = np.sum(proj ** 2) / (np.sum(noise ** 2) + eps)
    sisnr = 10 * np.log(ratio + eps) / np.log(10.0)
    return sisnr


def cal(path, permutation=False):
    mix_number = len(set([l.split('_')[0]
                          for l in os.listdir(path) if l[-3:] == 'wav']))
    print('num of mixed :', mix_number)
    SDR_sum = np.array([])
    SDRi_sum = np.array([])
    SISNRi_sum = np.array([])
    for idx in range(mix_number):
        pre_speech_channel = []
        aim_speech_channel = []
        mix_speech = []
        for l in sorted(os.listdir(path)):
            if l[-3:] != 'wav':
                continue
            if l.split('_')[0] == str(idx):
                if 'True_mix' in l:
                    mix_speech.append(sf.read(path + l)[0])
                if 'real' in l and 'noise' not in l:
                    aim_speech_channel.append(sf.read(path + l)[0])
                if 'pre' in l:
                    pre_speech_channel.append(sf.read(path + l)[0])

        assert len(aim_speech_channel) == len(pre_speech_channel)
        aim_speech_channel = np.array(aim_speech_channel)
        pre_speech_channel = np.array(pre_speech_channel)
        mix_speech = np.array(mix_speech)
        assert mix_speech.shape[0] == 1
        mix_speech = mix_speech[0]

        print(aim_speech_channel[:, :5])
        print(pre_speech_channel[:, :5])
        result = bss_eval_sources(
            aim_speech_channel, pre_speech_channel, compute_permutation=permutation)
        print(result)
        SDR_sum = np.append(SDR_sum, result[0])

        SDRi = cal_SDRi(aim_speech_channel, pre_speech_channel, mix_speech)
        print('SDRi:', SDRi)
        SDRi_sum = np.append(SDRi_sum, SDRi)
        SISNRi = cal_SISNRi(aim_speech_channel, pre_speech_channel, mix_speech)
        print('SI-SNRi', SISNRi)
        SISNRi_sum = np.append(SISNRi_sum, SISNRi)

    print('SDR_Aver for this batch:', SDR_sum.mean())
    print('SDRi_Aver for this batch:', SDRi_sum.mean())
    print('SISNRi_Aver for this batch:', SISNRi_sum.mean())
    return SDR_sum.mean(), SDRi_sum.mean(), SISNRi_sum.mean()


def cal_using_wav(batch_size, mix_speech, aim_speech, pre_speech, permutation=False):
    # bs * steps
    SDR_sum = np.array([])
    SDRi_sum = np.array([])
    SISNRi_sum = np.array([])
    for idx in range(batch_size):
        pre_speech_channel = pre_speech[idx]
        aim_speech_channel = aim_speech[idx]
        mix_speech_channel = mix_speech[idx]
        aim_speech_channel = np.array(aim_speech_channel.cpu().data)
        pre_speech_channel = np.array(pre_speech_channel.cpu().data)
        mix_speech_channel = np.array(mix_speech_channel.cpu().data)

        result = bss_eval_sources(
            aim_speech_channel, pre_speech_channel, compute_permutation=permutation)
        print(result)
        SDR_sum = np.append(SDR_sum, result[0])

        SDRi = result[0] - bss_eval_sources(aim_speech_channel,
                                            mix_speech_channel, compute_permutation=permutation)[0]
        print('SDRi:', SDRi)
        SDRi_sum = np.append(SDRi_sum, SDRi)
        SISNRi = cal_SISNRi(aim_speech_channel,
                            pre_speech_channel, mix_speech_channel)
        print('SI-SNRi', SISNRi)
        SISNRi_sum = np.append(SISNRi_sum, SISNRi)

    print('SDR_Aver for this batch:', SDR_sum.mean())
    print('SDRi_Aver for this batch:', SDRi_sum.mean())
    print('SISNRi_Aver for this batch:', SISNRi_sum.mean())
    return SDR_sum.mean(), SDRi_sum.mean(), SISNRi_sum.mean()
