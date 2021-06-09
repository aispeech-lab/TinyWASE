# coding=utf8
import os
import argparse
import time
import collections
import torch
from torch.autograd import Variable
import torch.utils.data
from tensorboardX import SummaryWriter
import numpy as np
from anybit import QuaOp
import utils
from data.preparedata import prepare_data

def test(model, config, opt, writer, logging, updates=0, mode='valid', bit=3):
    ss_loss_SUM = np.array([])
    SDR_SUM = np.array([])
    SDRi_SUM = np.array([])
    SISNRi_SUM = np.array([])

    accuracy_0_SUM = np.array([])
    f1_0_SUM = np.array([])
    accuracy_1_SUM = np.array([])
    f1_1_SUM = np.array([])
    accuracy_2_SUM = np.array([])
    f1_2_SUM = np.array([])
    accuracy_3_SUM = np.array([])
    f1_3_SUM = np.array([])

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
            logging('accuracy_0_aver_eval_epoch: {}'.format(accuracy_0_SUM.mean()))
            logging('f1_0_aver_eval_epoch: {}'.format(f1_0_SUM.mean()))
            logging('accuracy_1_aver_eval_epoch: {}'.format(accuracy_1_SUM.mean()))
            logging('f1_1_aver_eval_epoch: {}'.format(f1_1_SUM.mean()))
            logging('accuracy_2_aver_eval_epoch: {}'.format(accuracy_2_SUM.mean()))
            logging('f1_2_aver_eval_epoch: {}'.format(f1_2_SUM.mean()))
            logging('accuracy_3_aver_eval_epoch: {}'.format(accuracy_3_SUM.mean()))
            logging('f1_3_aver_eval_epoch: {}'.format(f1_3_SUM.mean()))
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
                predicted, oracle_endpoint, endpoint_0, endpoint_1, endpoint_2, endpoint_3 = model.module.test(
                    sorted_mix_wav, ref_wav, ref_wav_len, oracle_wav_endpoint)
            else:
                # predicted, oracle_endpoint, [endpoint_0, endpoint_1, endpoint_2, endpoint_3] = model.test(
                #     sorted_mix_wav, ref_wav, ref_wav_len, oracle_wav_endpoint)
                # # FIXME act
                predicted, oracle_endpoint, endpoint_0, endpoint_1, endpoint_2, endpoint_3 = model.test(
                    sorted_mix_wav, ref_wav, ref_wav_len, oracle_wav_endpoint)
        torch.cuda.empty_cache()

        predicted = predicted[:, :-1, :]
        sorted_aim_wav = sorted_aim_wav[:, :-1, :]
        aim_spk_list = [[aim_spk_list[0][0]]]

        if 1 and len(opt.gpus) > 1:
            ss_loss = model.module.separation_tas_loss(
                sorted_aim_wav, predicted, sorted_mix_wav_len)
            endpoint_loss_0, accuracy_0, precision_0, recall_0, f1_0 = model.module.endpoint_loss(
                endpoint_0, oracle_endpoint)
            endpoint_loss_1, accuracy_1, precision_1, recall_1, f1_1 = model.module.endpoint_loss(
                endpoint_1, oracle_endpoint)
            endpoint_loss_2, accuracy_2, precision_2, recall_2, f1_2 = model.module.endpoint_loss(
                endpoint_2, oracle_endpoint)
            endpoint_loss_3, accuracy_3, precision_3, recall_3, f1_3 = model.module.endpoint_loss(
                endpoint_3, oracle_endpoint)
        else:
            ss_loss = model.separation_tas_loss(
                sorted_aim_wav, predicted, sorted_mix_wav_len)
            endpoint_loss_0, accuracy_0, precision_0, recall_0, f1_0 = model.endpoint_loss(
                endpoint_0, oracle_endpoint)
            endpoint_loss_1, accuracy_1, precision_1, recall_1, f1_1 = model.endpoint_loss(
                endpoint_1, oracle_endpoint)
            endpoint_loss_2, accuracy_2, precision_2, recall_2, f1_2 = model.endpoint_loss(
                endpoint_2, oracle_endpoint)
            endpoint_loss_3, accuracy_3, precision_3, recall_3, f1_3 = model.endpoint_loss(
                endpoint_3, oracle_endpoint)
        endpoint_loss = torch.mean(torch.stack(
            (endpoint_loss_0, endpoint_loss_1, endpoint_loss_2, endpoint_loss_3), 0))

        ss_loss_SUM = np.append(ss_loss_SUM, ss_loss.cpu().item())
        accuracy_0_SUM = np.append(accuracy_0_SUM, accuracy_0.cpu().item())
        f1_0_SUM = np.append(f1_0_SUM, f1_0.cpu().item())
        accuracy_1_SUM = np.append(accuracy_1_SUM, accuracy_1.cpu().item())
        f1_1_SUM = np.append(f1_1_SUM, f1_1.cpu().item())
        accuracy_2_SUM = np.append(accuracy_2_SUM, accuracy_2.cpu().item())
        f1_2_SUM = np.append(f1_2_SUM, f1_2.cpu().item())
        accuracy_3_SUM = np.append(accuracy_3_SUM, accuracy_3.cpu().item())
        f1_3_SUM = np.append(f1_3_SUM, f1_3.cpu().item())

        predicted /= torch.max(torch.abs(predicted), dim=2, keepdim=True)[0]

        try:
            sdr_aver_batch, sdri_aver_batch, sisnri_aver_batch = utils.bss_test.cal_using_wav(
                config.test_batch_size, sorted_mix_wav, sorted_aim_wav, predicted)
            SDR_SUM = np.append(SDR_SUM, sdr_aver_batch)
            SDRi_SUM = np.append(SDRi_SUM, sdri_aver_batch)
            SISNRi_SUM = np.append(SISNRi_SUM, sisnri_aver_batch)
        except AssertionError as wrong_info:
            logging('Errors in calculating the SDR: %s' % wrong_info)


        logging('endpoint_loss_0: %f' % endpoint_loss_0.cpu().item())
        logging('accuracy_0: %f' % accuracy_0.cpu().item())
        logging('precision_0: %f' % precision_0.cpu().item())
        logging('recall_0: %f' % recall_0.cpu().item())
        logging('f1_0: %f' % f1_0.cpu().item())
        logging('endpoint_loss_1: %f' % endpoint_loss_1.cpu().item())
        logging('accuracy_1: %f' % accuracy_1.cpu().item())
        logging('precision_1: %f' % precision_1.cpu().item())
        logging('recall_1: %f' % recall_1.cpu().item())
        logging('f1_1: %f' % f1_1.cpu().item())
        logging('endpoint_loss_2: %f' % endpoint_loss_2.cpu().item())
        logging('accuracy_2: %f' % accuracy_2.cpu().item())
        logging('precision_2: %f' % precision_2.cpu().item())
        logging('recall_2: %f' % recall_2.cpu().item())
        logging('f1_2: %f' % f1_2.cpu().item())
        logging('endpoint_loss_3: %f' % endpoint_loss_3.cpu().item())
        logging('accuracy_3: %f' % accuracy_3.cpu().item())
        logging('precision_3: %f' % precision_3.cpu().item())
        logging('recall_3: %f' % recall_3.cpu().item())
        logging('f1_3: %f' % f1_3.cpu().item())
        logging('endpoint_loss: %f' % endpoint_loss.cpu().item())
        logging('SS loss: %f' % ss_loss.cpu().item())
        logging('SDR_aver_now: %f' % SDR_SUM.mean())
        logging('SDRi_aver_now: %f' % SDRi_SUM.mean())
        logging('SISNRi_aver_now: %f' % SISNRi_SUM.mean())

        batch_idx += 1

    writer.add_scalars('scalar/endpoint_loss',
                       {'endpoint_loss': endpoint_loss.cpu().item(), }, updates)
    writer.add_scalars('scalar/SDR', {'SDR_eval': SDR_SUM.mean(), }, updates)
    writer.add_scalars('scalar/SDRi', {'SDRi_eval': SDRi_SUM.mean(),}, updates)
    writer.add_scalars('scalar/SISNRi', {'SISNRi_eval': SISNRi_SUM.mean(),}, updates)
    writer.add_scalars('scalar/accuracy_0',
                       {'accuracy_0_eval': accuracy_0_SUM.mean(), }, updates)
    writer.add_scalars(
        'scalar/f1_0', {'f1_0_eval': f1_0_SUM.mean(), }, updates)
    writer.add_scalars('scalar/accuracy_1',
                       {'accuracy_1_eval': accuracy_1_SUM.mean(), }, updates)
    writer.add_scalars(
        'scalar/f1_1', {'f1_1_eval': f1_1_SUM.mean(), }, updates)
    writer.add_scalars('scalar/accuracy_2',
                       {'accuracy_2_eval': accuracy_2_SUM.mean(), }, updates)
    writer.add_scalars(
        'scalar/f1_2', {'f1_2_eval': f1_2_SUM.mean(), }, updates)
    writer.add_scalars('scalar/accuracy_3',
                       {'accuracy_3_eval': accuracy_3_SUM.mean(), }, updates)
    writer.add_scalars(
        'scalar/f1_3', {'f1_3_eval': f1_3_SUM.mean(), }, updates)

    score = {}
    score['SDR'] = SDR_SUM.mean()
    return score, ss_loss_SUM.mean()


if __name__ == '__main__':

    # config
    parser = argparse.ArgumentParser(description='train.py')
    parser.add_argument('-config', default='config.yaml', type=str,
                        help="config file")
    parser.add_argument('-gpus', default=[0], nargs='+', type=int,
                        help="Use CUDA on the listed devices.")
    parser.add_argument('-restore', default="TDAAv3_10_ak8.pt", type=str,
                        help="restore checkpoint")
    parser.add_argument('-seed', default=1234, type=int,
                        help="Random seed")
    parser.add_argument('-log', default='log_3x8_tiny', type=str,
                        help="log directory")
    parser.add_argument('-memory', default=False, type=bool,
                        help="memory efficiency")
    parser.add_argument('-score_fc', default='linear', type=str,
                        help="score function")
    parser.add_argument('-sharing', default=0, type=int, help='weight sharing')     


    opt = parser.parse_args()
    config = utils.util.read_config(opt.config)
    torch.manual_seed(opt.seed)
    torch.backends.cudnn.deterministic = True

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

    # logging module
    if not os.path.exists(config.log):
        os.mkdir(config.log)

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
        restore_path = os.path.join(log_path, opt.restore)
        checkpoints = torch.load(restore_path, map_location={'cuda:2': 'cuda:0'})
        if 'ss_model.encoder.weight' in checkpoints['model'].keys():
            print('Deleting ss_model.encoder.weight')
            checkpoints['model'].pop('ss_model.encoder.weight')
        bit = checkpoints['bit']
        alpha = checkpoints['alpha']
        beta = checkpoints['beta']
        qw_values = checkpoints['QW_values']
        QW_biases = checkpoints['QW_bias']

    # model
    print('building model...\n')

    if opt.sharing:
        from models.qwase_pshare import *
    else:
        from models.qwase import *
    model = wase_q(config, fre_size, frame_num,
        num_labels, use_cuda, opt.score_fc, QA_flag=True, ak=8)

    logging(repr(model) + "\n")
    print('loading checkpoint...\n', restore_path)
    model.load_state_dict(checkpoints['model'])


    if use_cuda:
        model.cuda()
    print('Quantization {}bit'.format(bit))
    print('QW_bias', QW_biases)
    print('QW_values {}'.format(qw_values))

    qua_op = QuaOp([model.ss_model.TCN], QW_biases, qw_values, initialize_biases=False)
    
    # parameter number
    param_count = 0
    for param in model.parameters():    
        param_count += param.view(-1).size()[0]
    logging('parameter number: %d\n' % param_count)

    voiceP_start_time, train_start_time, eval_start_timeonset_offset = time.time(), time.time(), time.time()
    total_voiceP_loss, total_ss_loss, total_loss = 0, 0, 0
    total_sample_num, total_correct = 0, 0
    scores = [[] for metric in config.METRIC]
    scores = collections.OrderedDict(zip(config.METRIC, scores))

    model.eval()

    # quantization
    print('alpha: ', alpha)
    print('beta: ', beta)
    print('len of alpha', len(alpha))
    print('len of beta: ', len(beta))
    print('bit', bit)
    qua_op.quantization(3000, alpha, beta, init=False, train_phase=False)

    test(model, config, opt, writer, logging, updates=0, mode='test', bit=bit)
