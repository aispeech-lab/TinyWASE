# <encoding:utf8>
"""
合成两说话人测试语音 on WSJ 数据集 by Jacoxu
# WSJ0: https://catalog.ldc.upenn.edu/docs/LDC93S6A/
# WSJ1: https://catalog.ldc.upenn.edu/docs/LDC94S13B/
"""

import pickle
import time
import resampy
import random
import numpy as np
import soundfile as sf
import os


random.seed(5678)


def split_train_eval_speakers(spk_train_path, spk_test_path):
    speaker_file_f = open('./spk_info.txt', 'r', encoding="utf-8")
    spk_train_f = open(spk_train_path, 'w', encoding='utf-8')
    spk_test_f = open(spk_test_path, 'w', encoding='utf-8')
    spk_list = []
    line_idx = 0
    for spk_info_line in speaker_file_f.readlines():
        if line_idx % 1000 == 0:
            print('Has processed %d lines' % line_idx)
        line_tokens = spk_info_line.split()
        if len(line_tokens) < 2:
            continue
        line_idx += 1
        spk_id = line_tokens[0] + ',' + line_tokens[2]
        spk_list.append(spk_id)
    speaker_file_f.close()
    random.shuffle(spk_list)
    line_idx = 0
    test_spk_num = 0
    train_spk_num = 0
    for spk_id in spk_list:
        line_idx += 1
        if line_idx % 4 == 0:
            spk_test_f.write(spk_id + '\n')
            test_spk_num += 1
        else:
            spk_train_f.write(spk_id + '\n')
            train_spk_num += 1
    spk_train_f.close()
    spk_test_f.close()
    print('Total speaker num:%d, where has train num:%d and test num:%d' %
          (len(spk_list), train_spk_num, test_spk_num))


def get_spk_set(spk_path, train=False):
    spk_set = set()
    spk_dict = dict()
    speaker_file_f = open(spk_path, 'r', encoding="utf-8")
    spk_idx = 0
    for spk_info_line in speaker_file_f.readlines():
        line_tokens = spk_info_line.strip().split(',')
        if len(line_tokens) < 1:
            continue
        spk_id = line_tokens[0]
        spk_set.add(spk_id)
        if train:
            spk_dict[spk_id] = spk_idx
        spk_idx += 1
    speaker_file_f.close()
    print('Loaded speaker set of %s, with %d speakers' %
          (spk_path, len(spk_set)))
    if train:
        if len(spk_set) != len(spk_dict):
            print('ERROR and STOP! spk_set len: %d is not equal with spk_dict len:%d' % (
                len(spk_set), len(spk_dict)))
            exit()
        return spk_set, spk_dict
    return spk_set


def complete_wsj_test_list(spk_test_path, spk_test_voiceP_path, data_path_prefix):
    spk_test_path_f = open(spk_test_path, 'r', encoding="utf-8")
    spk_test_voiceP_path_f = open(spk_test_voiceP_path, 'w', encoding='utf-8')
    line_idx = 0
    for spk_test_line in spk_test_path_f.readlines():
        if line_idx % 1000 == 0:
            print('Has processed %d lines' % line_idx)
        line_tokens = spk_test_line.split()
        if len(line_tokens) < 4:
            continue
        line_idx += 1
        spk1_wav_path = data_path_prefix + line_tokens[0]
        spk1_mix_dB = line_tokens[1]
        spk2_wav_path = data_path_prefix + line_tokens[2]
        spk2_mix_dB = line_tokens[3]
        spk_ref_folder = os.path.dirname(spk1_wav_path)
        spk_wav_file = random.choice(os.listdir(spk_ref_folder))
        spk_ref_file = os.path.join(spk_ref_folder, spk_wav_file)
        spk_test_voiceP_path_f.write(' '.join(
            [spk1_wav_path, spk1_mix_dB, spk2_wav_path, spk2_mix_dB, spk_ref_file, '\n']))
        spk_ref_folder = os.path.dirname(spk2_wav_path)
        spk_wav_file = random.choice(os.listdir(spk_ref_folder))
        spk_ref_file = os.path.join(spk_ref_folder, spk_wav_file)
        spk_test_voiceP_path_f.write(' '.join(
            [spk2_wav_path, spk2_mix_dB, spk1_wav_path, spk1_mix_dB, spk_ref_file, '\n']))
    spk_test_path_f.close()
    spk_test_voiceP_path_f.close()

    print('Total test num:%d' % line_idx)


class PrepareDataSamples(object):

    def __init__(self, config, check_len=False):
        print('Initialize prepare data object ...')
        self.config = config
        self.spk_train_set, self.spk_train_dict = get_spk_set(
            self.config['spk_train_path'], train=True)
        self.spk_test_set = get_spk_set(self.config['spk_test_path'])
        self.speaker_train_folder_dict = dict()
        self.speaker_train_wav_dict = dict()
        self.speaker_test_folder_list = []
        self.speaker_test_wav_dict = dict()
        self.train_number = 0
        self.test_number = 0
        self.test_sample_idx = 0
        if self.config['use_pickle']:
            self.save_to_pickle = not (os.path.exists(self.config['train_sample_pickle']) and
                                       os.path.exists(self.config['test_sample_pickle']))
        else:
            self.save_to_pickle = False
        self.onset = self.config['onset']
        self.offset = self.config['offset']
        self.speaker = self.config['speaker']

        for folder in self.config['speechWavFolderList']:
            for spk_folder in os.listdir(folder):
                if spk_folder in self.spk_train_set:
                    spk_idx = self.spk_train_dict[spk_folder]
                    spk_wav_list = []
                    for spk_wav_file in os.listdir(os.path.join(folder, spk_folder)):
                        wav_file = os.path.join(
                            folder, spk_folder, spk_wav_file)
                        if check_len:
                            wav, wav_rate = sf.read(wav_file)
                            if len(wav.shape) > 1:
                                wav = wav[:, 0]
                            if float(len(wav)) / wav_rate * self.config['rate'] > self.config['valid_wav_len_span'][0]:
                                spk_wav_list.append(wav_file)
                            else:
                                continue
                        else:
                            spk_wav_list.append(wav_file)
                        if self.save_to_pickle is True:
                            wav, wav_rate = sf.read(wav_file)
                            if len(wav.shape) > 1:
                                wav = wav[:, 0]
                            if wav_rate != self.config['rate']:
                                wav = resampy.resample(
                                    wav, wav_rate, self.config['rate'], filter='kaiser_best')
                            self.speaker_train_wav_dict[wav_file] = (
                                np.float16(wav), self.config['rate'])
                            print('wav_file:', wav_file)
                        self.train_number += 1
                    if len(spk_wav_list) < 2:
                        continue
                    self.speaker_train_folder_dict[spk_idx] = spk_wav_list
        print('Train sample number is: %d' % self.train_number)

        test_sample_file_f = open(
            self.config['spk_test_voiceP_path'], 'r', encoding="utf-8")
        for spk_info_line in test_sample_file_f.readlines():
            line_tokens = spk_info_line.strip().split(' ')
            if len(line_tokens) < 5:
                continue
            self.test_number += 1
            self.speaker_test_folder_list.append(line_tokens)
            if self.save_to_pickle is True:
                target_spk_clean_file, _, mask_spk_file, _, target_spk_ref_file = line_tokens
                for item in (target_spk_clean_file, mask_spk_file, target_spk_ref_file):
                    wav, wav_rate = sf.read(item)
                    if len(wav.shape) > 1:
                        wav = wav[:, 0]
                    if wav_rate != self.config['rate']:
                        wav = resampy.resample(
                            wav, wav_rate, self.config['rate'], filter='kaiser_best')
                    self.speaker_test_wav_dict[item] = (
                        np.float16(wav), self.config['rate'])
        print('Test sample number is: %d' % self.test_number)

        if self.config['use_pickle']:
            if self.save_to_pickle is True:
                print('save data object to pickle ...')
                train_pickle_file = open(
                    self.config['train_sample_pickle'], 'wb')
                pickle.dump(self.speaker_train_wav_dict, train_pickle_file)
                train_pickle_file.close()
                test_pickle_file = open(
                    self.config['test_sample_pickle'], 'wb')
                pickle.dump(self.speaker_test_wav_dict, test_pickle_file)
                test_pickle_file.close()
            else:
                with open(self.config['train_sample_pickle'], 'rb') as train_pickle_file:
                    self.speaker_train_wav_dict = pickle.load(
                        train_pickle_file)
                with open(self.config['test_sample_pickle'], 'rb') as test_pickle_file:
                    self.speaker_test_wav_dict = pickle.load(test_pickle_file)
            print('speaker_train_wav_dict size:',
                  len(self.speaker_train_wav_dict))
            print('speaker_test_wav_dict size:',
                  len(self.speaker_test_wav_dict))

    def get_samples_for_voiceP(self, batch_size, train=True):
        sample_voice_list = []
        sample_ss_list = []
        for sample_idx in range(batch_size):
            if train:
                target_spk_id, mask_spk_id = random.sample(
                    self.speaker_train_folder_dict.keys(), 2)
                target_spk_clean_file, target_spk_ref_file = random.sample(
                    self.speaker_train_folder_dict[target_spk_id], 2)
                mask_spk_file = random.choice(
                    self.speaker_train_folder_dict[mask_spk_id])
                db_rate = np.random.uniform(
                    self.config['masking_dB_span'][0], self.config['masking_dB_span'][1], 1)
                ratio_target = 10 ** (db_rate / 20.0)
                ratio_mask = 10 ** (-1 * db_rate / 20.0)
            else:
                target_spk_clean_file, target_mix_dB, mask_spk_file, mask_mix_dB, target_spk_ref_file \
                    = self.speaker_test_folder_list[self.test_sample_idx % self.test_number]
                self.test_sample_idx += 1
                target_spk_id = 'None1'
                mask_spk_id = 'None2'
                ratio_target = 10 ** (float(target_mix_dB) / 20.0)
                ratio_mask = 10 ** (float(mask_mix_dB) / 20.0)

            # no extra requirement to wav length when using LSTM as voiceprint module
            target_spk_ref_wav, wav_rate = self.read_data(
                target_spk_ref_file, train)
            target_spk_ref_wav = self.preprocess_data(
                target_spk_ref_wav, wav_rate)

            target_spk_clean_wav, wav_rate = self.read_data(
                target_spk_clean_file, train)
            target_spk_clean_wav = self.preprocess_data(
                target_spk_clean_wav, wav_rate)
            target_spk_clean_wav = list(target_spk_clean_wav)
            if train:
                target_spk_clean_wav = self.length_control(
                    target_spk_clean_wav)
            else:
                pass
            if train and not (self.onset or self.offset):
                target_spk_clean_wav = self.random_shift_data(
                    target_spk_clean_wav)
            target_spk_clean_wav = np.array(target_spk_clean_wav)

            mask_spk_wav, wav_rate = self.read_data(mask_spk_file, train)
            mask_spk_wav = self.preprocess_data(mask_spk_wav, wav_rate)
            mask_spk_wav = list(mask_spk_wav)
            if train:
                mask_spk_wav = self.length_control(mask_spk_wav)
            else:
                longer_wav_len = max(len(mask_spk_wav), len(
                    list(target_spk_clean_wav)))
                mask_spk_wav.extend(
                    np.zeros(longer_wav_len - len(mask_spk_wav)))
                target_spk_clean_wav = list(target_spk_clean_wav)
                target_spk_clean_wav.extend(
                    np.zeros(longer_wav_len - len(target_spk_clean_wav)))
                target_spk_clean_wav = np.array(target_spk_clean_wav)
            if train and not (self.onset or self.offset):
                mask_spk_wav = self.random_shift_data(mask_spk_wav)
            mask_spk_wav = np.array(mask_spk_wav)

            target_spk_clean_wav = ratio_target * target_spk_clean_wav
            mask_spk_wav = ratio_mask * mask_spk_wav
            wav_mix = target_spk_clean_wav + mask_spk_wav

            # (target_ref)
            sample_voice_list.append((target_spk_ref_wav))
            # (mix, target_clean, target_id, mask_wav, mask_id)
            sample_ss_list.append(
                (wav_mix, target_spk_clean_wav, target_spk_id, mask_spk_wav, mask_spk_id))
        return sample_voice_list, sample_ss_list

    def get_samples_for_voiceP_3_speakers(self, batch_size, train=True):
        sample_voice_list = []
        sample_ss_list = []
        for sample_idx in range(batch_size):
            if train:
                spk_list = random.sample(
                    self.speaker_train_folder_dict.keys(), self.speaker)
                target_spk_id = spk_list[0]
                mask_spk_id = spk_list[1]  #
                target_spk_ref_file = random.choice(
                    self.speaker_train_folder_dict[target_spk_id])
                spk_file_list = []
                for spk in spk_list:
                    spk_file_list.append(random.choice(
                        self.speaker_train_folder_dict[spk]))
                db_rate = np.random.uniform(
                    self.config['masking_dB_span'][0], self.config['masking_dB_span'][1], 1)
                ratio_list = [10 ** (db_rate / 20.0), 10 **
                              (-1 * db_rate / 20.0), 1]
                random.shuffle(ratio_list)
            else:
                target_spk_clean_file, target_mix_dB, mask_spk_file, mask_mix_dB, target_spk_ref_file \
                    = self.speaker_test_folder_list[self.test_sample_idx % self.test_number]
                self.test_sample_idx += 1
                target_spk_id = 'None1'
                mask_spk_id = 'None2'
                spk_file_list = [target_spk_clean_file, mask_spk_file]
                ratio_target = 10 ** (float(target_mix_dB) / 20.0)
                ratio_mask = 10 ** (float(mask_mix_dB) / 20.0)
                ratio_list = [ratio_target, ratio_mask]

            # no extra requirement to wav length when using LSTM as voiceprint module
            target_spk_ref_wav, wav_rate = self.read_data(
                target_spk_ref_file, train)
            target_spk_ref_wav = self.preprocess_data(
                target_spk_ref_wav, wav_rate)

            spk_wav_list = []
            for spk_file in spk_file_list:
                spk_wav, wav_rate = self.read_data(spk_file, train)
                spk_wav = self.preprocess_data(spk_wav, wav_rate)
                spk_wav = list(spk_wav)
                if train:
                    spk_wav = self.length_control(spk_wav)
                spk_wav_list.append(spk_wav)
            if not train:
                longer_wav_len = max(len(spk_wav) for spk_wav in spk_wav_list)
                for spk_wav in spk_wav_list:
                    spk_wav.extend(np.zeros(longer_wav_len - len(spk_wav)))
            wav_mix = 0
            for spk_wav, ratio in zip(spk_wav_list, ratio_list):
                wav_mix += np.array(spk_wav) * ratio
            target_spk_clean_wav = np.array(spk_wav_list[0]) * ratio_list[0]
            mask_spk_wav = wav_mix - target_spk_clean_wav

            # (target_ref)
            sample_voice_list.append((target_spk_ref_wav))
            # (mix, target_clean, target_id, mask_wav, mask_id)
            sample_ss_list.append(
                (wav_mix, target_spk_clean_wav, target_spk_id, mask_spk_wav, mask_spk_id))
        return sample_voice_list, sample_ss_list

    def read_data(self, spk_file, train):
        if self.config['use_pickle']:
            try:
                if train:
                    spk_wav, wav_rate = self.speaker_train_wav_dict[spk_file]
                else:
                    spk_wav, wav_rate = self.speaker_test_wav_dict[spk_file]
                spk_wav = np.float32(spk_wav)
            except KeyError:
                spk_wav, wav_rate = sf.read(spk_file)
        else:
            spk_wav, wav_rate = sf.read(spk_file)
        return spk_wav, wav_rate

    def preprocess_data(self, wav, wav_rate):
        if len(wav.shape) > 1:
            wav = wav[:, 0]
        if wav_rate != self.config['rate']:
            wav = resampy.resample(
                wav, wav_rate, self.config['rate'], filter='kaiser_best')
        wav -= np.mean(wav)
        wav /= np.max(np.abs(wav)) + np.spacing(1)
        return wav

    def random_shift_data(self, wav):
        random_shift = random.choice(range(self.config['mix_wav_len']))
        wav = wav[random_shift:] + wav[:random_shift]
        return wav

    def length_control(self, wav):
        if len(wav) > self.config['valid_wav_len_span'][1]:
            # guarantee the existence of onset/offset
            if self.offset is True:
                random_cut = random.choice(
                    range(int(self.config['rate'] * 3.2), int(self.config['rate'] * 3.9)))
                wav = wav[:random_cut]
                wav.extend(np.zeros(self.config['mix_wav_len'] - len(wav)))
            else:
                wav = wav[:self.config['valid_wav_len_span'][1]]
        else:
            wav.extend(np.zeros(self.config['mix_wav_len'] - len(wav)))
        return wav


if __name__ == "__main__":
    print('Test 001')
    """
    Instructions:
    # split_train_eval_speakers(spk_train_path, spk_test_path)
    # data_path_prefix = '/data/xujiaming_data/WSJ/'
    # spk_test_path = './spk_test_WSJ.lst'
    # spk_test_voiceP_path = './mix_2_spk_voiceP_tt_WSJ.txt'
    # complete_wsj_test_list(spk_test_path, spk_test_voiceP_path, data_path_prefix)
    Step001, define the parameters in config
    Step002, instantiate PrepareDataSamples
    Step003, obtain the training samples or evaluating samples by get_samples_for_voiceP
             train_sample_voice_list: (target_ref)
             train_sample_ss_list: (mix, target_clean, target_id, mask_wav, mask_id)
    """
    config = dict()
    config['speechWavFolderList'] = ['/home/aa/WSJ/wsj0/si_tr_s/']
    config['spk_test_voiceP_path'] = './mix_2_spk_voiceP_tt_WSJ_127.txt'
    config['spk_train_path'] = './spk_train_WSJ.lst'
    config['spk_test_path'] = './spk_test_WSJ.lst'
    config['train_sample_pickle'] = '/data1/haoyunzhe/interspeech/dataset/wsj0_pickle/train_sample.pickle'
    config['test_sample_pickle'] = '/data1/haoyunzhe/interspeech/dataset/wsj0_pickle/test_sample.pickle'
    config['rate'] = 8000
    config['batch_size'] = 16
    config['eval_num'] = 6
    config['mix_wav_len'] = 4 * config['rate']
    config['valid_wav_len_span'] = [1 * config['rate'], 4 * config['rate']]
    config['masking_dB_span'] = [-2.5, 2.5]
    config['noise_dB_span'] = [-15, -5]
    config['use_pickle'] = True
    config['onset'] = True
    config['offset'] = True
    config['noise'] = True
    config['speaker'] = 3
    prepareDataSamples = PrepareDataSamples(config, check_len=False)
    train_sample_voice_list, train_sample_ss_list = prepareDataSamples.get_samples_for_voiceP(
        config['batch_size'], train=True)
    test_sample_voice_list, test_sample_ss_list = prepareDataSamples.get_samples_for_voiceP(
        config['eval_num'], train=False)
    train_sample_voice_list, train_sample_ss_list = prepareDataSamples.get_samples_for_voiceP_3_speakers(
        config['batch_size'], train=True)
    test_sample_voice_list, test_sample_ss_list = prepareDataSamples.get_samples_for_voiceP_3_speakers(
        config['eval_num'], train=False)
