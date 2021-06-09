# coding=utf8
import os
import argparse
import time
import collections
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data
from tensorboardX import SummaryWriter
import numpy as np

import models
import utils
from data.preparedata import prepare_data

def test(model, config, opt, writer, logging, updates=0, mode='valid'):
    SDR_SUM = np.array([])
    SDRi_SUM = np.array([])
    SISNRi_SUM = np.array([])

    logging('Test or valid: %s' % mode)
    eval_data_gen = prepare_data('once', mode)

    batch_idx = 0
    while True:
        logging('-' * 30)
        eval_data = next(eval_data_gen)
        if eval_data is False:
            logging('SDR_aver_eval_epoch: %f' % SDR_SUM.mean())
            logging('SDRi_aver_eval_epoch: %f' % SDRi_SUM.mean())
            logging('SISNRi_aver_eval_epoch: %f' % SISNRi_SUM.mean())

            break

        aim_spk_list = eval_data['batch_order']
        oracle_wav_endpoint = torch.tensor(eval_data['oracle_wav_endpoint'])
        ref_wav = Variable(torch.tensor(eval_data['ref_wav']))
        ref_wav_len = Variable(torch.tensor(eval_data['ref_wav_length']))
        sorted_mix_wav, sorted_mix_wav_len, sorted_aim_wav = eval_data['wav_zip']
        sorted_mix_wav = torch.tensor(sorted_mix_wav)
        sorted_mix_wav_len = torch.from_numpy(sorted_mix_wav_len)
        sorted_aim_wav = torch.tensor(sorted_aim_wav)

        if config.use_cuda:
            oracle_wav_endpoint = oracle_wav_endpoint.cuda().float()
            ref_wav = ref_wav.cuda().float()
            ref_wav_len = ref_wav_len.cuda().float()
            sorted_mix_wav = sorted_mix_wav.cuda().float()
            sorted_mix_wav_len = sorted_mix_wav_len.cuda()
            sorted_aim_wav = sorted_aim_wav.cuda().float()

        with torch.no_grad():
            if 1 and len(opt.gpus) > 1:
                predicted, oracle_endpoint, endpoint_0, endpoint_1, endpoint_2, endpoint_3 = model.module.test(sorted_mix_wav, ref_wav, ref_wav_len, oracle_wav_endpoint)
            else:
                predicted, oracle_endpoint, endpoint_0, endpoint_1, endpoint_2, endpoint_3 = model.test(sorted_mix_wav, ref_wav, ref_wav_len, oracle_wav_endpoint)
        torch.cuda.empty_cache()

        predicted = predicted[:, :-1, :]
        sorted_aim_wav = sorted_aim_wav[:, :-1, :]
        aim_spk_list = [[aim_spk_list[0][0]]]


        predicted /= torch.max(torch.abs(predicted), dim=2, keepdim=True)[0]

        try:
            sdr_aver_batch, sdri_aver_batch, sisnri_aver_batch = utils.bss_test.cal_using_wav(
                config.test_batch_size, sorted_mix_wav, sorted_aim_wav, predicted)
            SDR_SUM = np.append(SDR_SUM, sdr_aver_batch)
            SDRi_SUM = np.append(SDRi_SUM, sdri_aver_batch)
            SISNRi_SUM = np.append(SISNRi_SUM, sisnri_aver_batch)
        except AssertionError as wrong_info:
            logging('Errors in calculating the SDR: %s' % wrong_info)

        logging('SDR_aver_now: %f' % SDR_SUM.mean())
        logging('SDRi_aver_now: %f' % SDRi_SUM.mean())
        logging('SISNRi_aver_now: %f' % SISNRi_SUM.mean())

        batch_idx += 1

    writer.add_scalars('scalar/SDR', {'SDR_eval': SDR_SUM.mean(), }, updates)
    writer.add_scalars('scalar/SDRi', {'SDRi_eval': SDRi_SUM.mean(),}, updates)
    writer.add_scalars('scalar/SISNRi', {'SISNRi_eval': SISNRi_SUM.mean(),}, updates)
        
    score = {}
    score['SDR'] = SDR_SUM.mean()
    return score, None


def modify_checkpoints(checkpoints):
    if 'ss_model.encoder.weight' in checkpoints['model'].keys():
        print('Deleting ss_model.encoder.weight')
        checkpoints['model'].pop('ss_model.encoder.weight')
        print('Changing the model keys!')
        checkpoints['model']['ss_model.TCN.output_act.weight'] = checkpoints['model'].pop('ss_model.TCN.output.0.weight')
        checkpoints['model']['ss_model.TCN.output_conv.weight'] = checkpoints['model'].pop('ss_model.TCN.output.1.weight')
        checkpoints['model']['ss_model.TCN.output_conv.bias'] = checkpoints['model'].pop('ss_model.TCN.output.1.bias')
    return checkpoints


if __name__ == '__main__':
    # config
    parser = argparse.ArgumentParser(description='train.py')
    parser.add_argument('-config', default='config.yaml', type=str,
                        help="config file")
    parser.add_argument('-gpus', default=[0], nargs='+', type=int,
                        help="Use CUDA on the listed devices.")
    parser.add_argument('-restore', default="TDAAv3_10.pt", type=str,
                        help="restore checkpoint")
    parser.add_argument('-seed', default=1234, type=int,
                        help="Random seed")
    parser.add_argument('-sharing', default=0, type=int, help='weight sharing')
    parser.add_argument('-log', default='log_3x8', type=str,
                        help="log directory")
    parser.add_argument('-memory', default=False, type=bool,
                        help="memory efficiency")
    parser.add_argument('-score_fc', default='linear', type=str,
                        help="score function")

    opt = parser.parse_args()
    config = utils.util.read_config(opt.config)
    torch.manual_seed(opt.seed)
    torch.backends.cudnn.deterministic = True

    # logging module
    if not os.path.exists(config.log):
        os.mkdir(config.log)
    if opt.log == '':
        log_path = config.log + utils.util.format_time(time.localtime()) + '/'
    else:
        log_path = config.log + opt.log + '/'
    if not os.path.exists(log_path):
        os.mkdir(log_path)
    print('log_path:', log_path)
    writer = SummaryWriter(log_path)
    logging = utils.util.logging(log_path + 'log.txt')
    logging_csv = utils.util.logging_csv(log_path + 'record.csv')
    for k, v in config.items():
        logging("%s:\t%s\n" % (str(k), str(v)))
    logging("\n")

    # checkpoint
    if opt.restore:
        print('loading checkpoint...\n', opt.restore)
        restore_path = os.path.join(log_path, opt.restore)
        checkpoints = torch.load(
            restore_path, map_location={'cuda:2': 'cuda:0'})
        checkpoints = modify_checkpoints(checkpoints)

    # cuda
    use_cuda = torch.cuda.is_available() and len(opt.gpus) > 0
    config.use_cuda = use_cuda
    if use_cuda:
        torch.cuda.set_device(opt.gpus[0])
        torch.cuda.manual_seed(opt.seed)
    print("use_cuda:", use_cuda)

    # load the global statistic of the data
    print('loading data...\n')
    start_time = time.time()

    # import parameters in the dataset
    spk_global_gen = prepare_data(mode='global')
    global_para = next(spk_global_gen)

    spk_list = global_para['all_spk']  # list of all speakers
    dict_spk2idx = global_para['dict_spk_to_idx']
    dict_idx2spk = global_para['dict_idx_to_spk']
    fre_size = global_para['num_fre']  # frequency size
    frame_num = global_para['num_frames']  # frame length
    spk_num = global_para['spk_num']  # speaker number
    batch_num = global_para['batch_num']  # batch number in a epoch

    config.fre_size = fre_size
    config.frame_num = frame_num
    num_labels = len(spk_list)
    del spk_global_gen
    print('loading the global setting cost: %.3f' % (time.time() - start_time))

    # model
    print('building model...\n')
    model = models.wase(config, fre_size, frame_num,
                        num_labels, use_cuda, opt.score_fc, sharing=opt.sharing)
    if opt.restore:
        model.load_state_dict(checkpoints['model'])
    if use_cuda:
        model.cuda()
    if len(opt.gpus) > 1:
        model = nn.DataParallel(model, device_ids=opt.gpus, dim=0)

    logging(repr(model) + "\n")

    # parameter number
    param_count = 0
    for param in model.parameters():
        param_count += param.view(-1).size()[0]
    logging('parameter number: %d\n' % param_count)

    voiceP_start_time, train_start_time, eval_start_time = time.time(), time.time(), time.time()
    total_voiceP_loss, total_ss_loss, total_loss = 0, 0, 0
    total_sample_num, total_correct = 0, 0
    scores = [[] for metric in config.METRIC]
    scores = collections.OrderedDict(zip(config.METRIC, scores))

    model.eval()

    test(model, config, opt, writer, logging, updates=0, mode='test')
