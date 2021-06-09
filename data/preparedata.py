# coding=utf8
import numpy as np
import librosa
import argparse
import utils
from data.wsj.prepareDataOnWSJ import PrepareDataSamples
parser = argparse.ArgumentParser(description='predata scripts.')
parser.add_argument('-config', default='config.yaml', type=str,
                    help="config file")
opt = parser.parse_args()
config = utils.util.read_config(opt.config)

data_config = dict()
data_config['speechWavFolderList'] = ['/mnt/lustre/xushuang4/dataset/WSJ/wsj0/si_tr_s/']
data_config['spk_test_voiceP_path'] = './data/wsj/mix_2_spk_voiceP_tt_WSJ_dellnode.txt'
data_config['spk_train_path'] = './data/wsj/spk_train_WSJ.lst'
data_config['spk_test_path'] = './data/wsj/spk_test_WSJ.lst'
data_config['train_sample_pickle'] = '/mnt/lustre/xushuang4/haoyunzhe/Documents/INTERSPEECH2020/dataset/wsj0_pickle/train_sample.pickle'
data_config['test_sample_pickle'] = '/mnt/lustre/xushuang4/haoyunzhe/Documents/INTERSPEECH2020/dataset/wsj0_pickle/test_sample.pickle'

data_config['rate'] = config.FRAME_RATE
data_config['batch_size'] = config.batch_size
data_config['test_batch_size'] = config.test_batch_size
data_config['mix_wav_len'] = config.MAX_LEN  # second
data_config['valid_wav_len_span'] = [
    1 * data_config['rate'], int(config.MAX_LEN)]
data_config['masking_dB_span'] = [-2.5, 2.5]
data_config['noise_dB_span'] = [-15, -5]
data_config['use_pickle'] = config.PICKLE
data_config['onset'] = config.ONSET
data_config['offset'] = config.OFFSET
data_config['noise'] = config.NOISE
data_config['speaker'] = config.SPEAKER
prepareDataSamples = PrepareDataSamples(data_config, check_len=False)



def get_energy_order(spk_feas_dict_list):
    order = []
    for one_line in spk_feas_dict_list:
        dd = sorted(one_line.items(), key=lambda d: d[1].sum(), reverse=True)
        dd = [d[0] for d in dd]
        order.append(dd)
    return order


def _collate_fn(mix_data, source_data, raw_tgt=None):
    """
    Args:
        batch: list, len(batch) = 1. See AudioDataset.__getitem__()
    Returns:
        mixtures_pad: B x T, torch.Tensor
        ilens : B, torch.Tentor
        sources_pad: B x C x T, torch.Tensor
    """
    mixtures = mix_data
    if raw_tgt is None:
        raw_tgt = [sorted(spk.keys()) for spk in source_data]
    sources = []
    for each_feas, each_line in zip(source_data, raw_tgt):
        sources.append(np.stack([each_feas[spk] for spk in each_line]))
    sources = np.array(sources)
    ilens = np.array([mix.shape[0] for mix in mixtures])
    return mixtures, ilens, sources


def vad_merge(w):
    intervals = librosa.effects.split(w, top_db=20)
    temp = list()
    for s, e in intervals:
        temp.append(w[s:e])
    return np.concatenate(temp, axis=None)


def get_oracle_wav_endpoint(w, offset=False):
    intervals = librosa.effects.split(w, top_db=20)
    oracle_wav_endpoint = np.zeros_like(w)
    for s, e in intervals:
        if offset is False:
            oracle_wav_endpoint[s:] = 1
            break
        else:
            # oracle_wav_endpoint[s:e] = 1
            oracle_wav_endpoint[intervals[0][0]:intervals[-1][-1]] = 1
            break
    return oracle_wav_endpoint


def get_mel(y):
    mel_basis = librosa.filters.mel(sr=8000, n_fft=512, n_mels=40)
    y = librosa.core.stft(y=y, n_fft=512, hop_length=80,
                          win_length=200, window='hann')
    magnitudes = np.abs(y) ** 2
    mel = np.log10(np.dot(mel_basis, magnitudes) + 1e-6)
    return mel


def get_stft_feas(wav, win, stride, feas_type="stft"):
    if feas_type == "stft":
        return np.transpose(np.abs(librosa.core.spectrum.stft(wav, win, stride)))
    elif feas_type == "phase":
        return np.transpose(np.angle(librosa.core.spectrum.stft(wav, win, stride)))
    elif feas_type == "complex":
        return np.transpose(librosa.core.spectrum.stft(wav, win, stride))


def prepare_data(mode, train_or_test='train'):
    if train_or_test == 'train':
        batch_size = data_config['batch_size']
        number_samples_all = prepareDataSamples.train_number
    elif train_or_test == 'valid':
        batch_size = data_config['test_batch_size']
        number_samples_all = 200 
    elif train_or_test == 'test':
        batch_size = data_config['test_batch_size']
        number_samples_all = prepareDataSamples.test_number
    batch_total = int(number_samples_all / batch_size)
    for batch_num in range(batch_total + 1):
        if batch_num == batch_total:
            print('ends here')
            yield False
        print('current batch num:', batch_num, ' of total: ', batch_total)
        if train_or_test == 'train':
            sample_voice_list, sample_ss_list = prepareDataSamples.get_samples_for_voiceP(
                batch_size, train=True)
        elif train_or_test == 'valid' or train_or_test == 'test':
            sample_voice_list, sample_ss_list = prepareDataSamples.get_samples_for_voiceP(
                batch_size, train=False)

        mix_wav_list = []
        aim_wav_list = []
        aim_feas_list = []
        aim_spk = []
        oracle_wav_endpoint_list = []
        mask_wav_list = []
        ref_wav_list = []
        ref_wav_length_list = []
        spk_wav_dict_list = []
        spk_feas_dict_list = []
        sorted_ss_aim_spk_list = []
        for batch_idx in range(batch_size):
            spk_wav_dict_this_sample = {}
            spk_feas_dict_this_sample = {}
            sorted_ss_aim_spk = []
            mix_wav, aim_wav, target_id, mask_wav, mask_id = sample_ss_list[batch_idx]
            target_ref = sample_voice_list[batch_idx]

            aim_spk.append([target_id])
            mix_wav_list.append(mix_wav)

            aim_wav_list.append(aim_wav)
            oracle_wav_endpoint = get_oracle_wav_endpoint(
                aim_wav, data_config['offset'])
            oracle_wav_endpoint_list.append(oracle_wav_endpoint)

            mask_wav_list.append(mask_wav)

            if config.VAD:
                target_ref = vad_merge(target_ref)
            if len(target_ref) % config.FRAME_LENGTH > 0:
                target_ref = target_ref[:-
                                        (len(target_ref) % config.FRAME_LENGTH)]
            ref_wav_list.append(target_ref)
            ref_wav_length_list.append(len(target_ref))

            spk_wav_dict_this_sample[target_id] = aim_wav
            spk_wav_dict_this_sample[mask_id] = mask_wav
            sorted_ss_aim_spk.append(target_id)
            sorted_ss_aim_spk.append(mask_id)
            sorted_ss_aim_spk_list.append(sorted_ss_aim_spk)

            spk_wav_dict_list.append(spk_wav_dict_this_sample)

        ref_wav_length_max = np.max(np.array(ref_wav_length_list))
        for batch_idx in range(batch_size):
            wav_length = ref_wav_length_list[batch_idx]
            if wav_length < ref_wav_length_max:  # some corpus has 80% silence
                ref_wav_list[batch_idx] = np.hstack(
                    (ref_wav_list[batch_idx], np.zeros((ref_wav_length_max - wav_length))))

        batch_order = sorted_ss_aim_spk_list
        if mode == 'global':
            all_spk = sorted(prepareDataSamples.spk_train_set)
            all_spk_test = sorted(prepareDataSamples.spk_test_set)
            dict_spk_to_idx = {spk: idx for idx, spk in enumerate(all_spk)}
            dict_idx_to_spk = {idx: spk for idx, spk in enumerate(all_spk)}
            yield {'all_spk': all_spk,
                   'dict_spk_to_idx': dict_spk_to_idx,
                   'dict_idx_to_spk': dict_idx_to_spk,
                   'num_fre': 129,
                   'num_frames': 100,
                   'spk_num': len(all_spk),
                   'batch_num': batch_total
                   }
        elif mode == 'once':
            yield {'mix_wav': mix_wav_list,
                   'aim_wav': aim_wav_list,
                   'aim_spk': aim_spk,
                   'oracle_wav_endpoint': oracle_wav_endpoint_list,
                   'mask_wav': mask_wav_list,
                   'ref_wav': ref_wav_list,
                   'ref_wav_length': ref_wav_length_list,
                   'spk_wav_dict_list': spk_wav_dict_list,
                   'batch_order': batch_order,
                   'wav_zip': _collate_fn(mix_wav_list, spk_wav_dict_list, batch_order),
                   'wav_info': [target_id, mask_id]
                   }
