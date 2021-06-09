# coding=utf8
from eval_wase import test
from data.preparedata import prepare_data
import utils
import models
import numpy as np
from tensorboardX import SummaryWriter
from torch.autograd import Variable
import torch.nn as nn
import torch
import warnings
import collections
import time
import argparse
import os


# config
parser = argparse.ArgumentParser(description='train.py')
parser.add_argument('-config', default='config.yaml', type=str,
                    help="config file")
parser.add_argument('-gpus', default=[0,1,2,3,4,5,6,7], nargs='+', type=int,
                    help="Use CUDA on the listed devices.")
# parser.add_argument('-pretrained', default="./log/log_3x8/TDAAv3_10.pt", type=str,
parser.add_argument('-pretrained', default=None, type=str,
                    help="restore checkpoint")
# parser.add_argument('-restore', default='TDAAv3_10.pt', type=str,
parser.add_argument('-restore', default=None, type=str,
                    help="restore checkpoint")
parser.add_argument('-seed', default=1234, type=int, # 1234
                    help="Random seed")
parser.add_argument('-log', default='log_3x8_pshare', type=str,
                    help="log directory")
parser.add_argument('-memory', default=False, type=bool,
                    help="memory efficiency")
parser.add_argument('-score_fc', default='linear', type=str,
                    help="score function")
parser.add_argument('-sharing', default=0, type=int,
                    help="weight sharing")

opt = parser.parse_args()
config = utils.util.read_config(opt.config)
torch.manual_seed(opt.seed)
torch.backends.cudnn.deterministic = True

# checkpoint
if opt.restore:
    print('loading checkpoint...\n', opt.restore)
    log_path = config.log + opt.log + '/'
    ckpt_path = os.path.join(log_path, opt.restore)
    checkpoints = torch.load(ckpt_path, map_location={'cuda:2': 'cuda:0'})

# cuda
use_cuda = torch.cuda.is_available() and len(opt.gpus) > 0
config.use_cuda = use_cuda
if use_cuda:
    torch.cuda.set_device(opt.gpus[0])
    torch.cuda.manual_seed(opt.seed)
print("use cuda:", use_cuda)

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
                    num_labels, use_cuda, opt.score_fc, teacher=False, sharing=opt.sharing)

if opt.pretrained:
    print('loading checkpoint from pretrained model...', opt.pretrained)
    teacher_checkpoint = torch.load(opt.pretrained)

    voiceprint_checkpoint = {k:v for k, v in teacher_checkpoint['model'].items()
        if (('ref_encoder' in k) or ('voiceprint_encoder' in k) or ('linear.' in k))}
    print('voiceprint network keys', voiceprint_checkpoint.keys())
    model.load_state_dict(voiceprint_checkpoint, strict=False)
    print('Loading the voiceprint network from teacher model!')

if opt.restore:
    model.load_state_dict(checkpoints['model'])

if use_cuda:
    model.cuda()
if len(opt.gpus) > 1:
    model = nn.DataParallel(model, device_ids=opt.gpus, dim=0)

# optimizer
optim = utils.optims.Optim(
    config.optim, config.learning_rate, config.max_grad_norm)
optim.set_parameters(model.parameters())

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
logging("\n")
for k, v in config.items():
    logging("%s:\t\t%s" % (str(k), str(v)))
logging("\n")
logging(repr(model) + "\n")

# parameter number
param_count = 0
for param in model.parameters():
    param_count += param.view(-1).size()[0]
logging('parameter number: %d\n' % param_count)

if opt.restore:
    updates = checkpoints['updates']
else:
    updates = 0

voiceP_start_time, train_start_time, eval_start_time = time.time(), time.time(), time.time()
total_voiceP_loss, total_ss_loss, total_loss = 0, 0, 0
total_sample_num, total_correct = 0, 0
scores = [[] for metric in config.METRIC]
scores = collections.OrderedDict(zip(config.METRIC, scores))


def train(epoch):
    global voiceP_start_time, train_start_time, eval_start_time
    global updates, total_voiceP_loss, total_ss_loss, total_loss, total_sample_num, total_correct
    model.train()
    SDR_SUM = np.array([])
    SDRi_SUM = np.array([])
    SISNRi_SUM = np.array([])

    logging("Epoch %g begin..." % epoch)
    logging("Decaying learning rate to %g" %
            optim.optimizer.param_groups[0]['lr'])

    train_data_gen = prepare_data('once', 'train')
    while True:
        updates_start_time = time.time()
        train_data = next(train_data_gen)
        if train_data is False:
            logging('SDR_aver_epoch: %f' % SDR_SUM.mean())
            logging('SDRi_aver_epoch: %f' % SDRi_SUM.mean())
            logging('SISNRi_aver_epoch: %f' % SISNRi_SUM.mean())
            logging('training epoch %d ends' % epoch)
            logging('-' * 30)
            break

        voiceP_aim_spk_list = train_data['aim_spk']
        oracle_wav_endpoint = torch.tensor(train_data['oracle_wav_endpoint'])
        aim_spk_list = train_data['batch_order']
        ref_wav = Variable(torch.tensor(train_data['ref_wav']))
        ref_wav_len = Variable(torch.tensor(train_data['ref_wav_length']))

        sorted_mix_wav, sorted_mix_wav_len, sorted_aim_wav = train_data['wav_zip']
        sorted_mix_wav = torch.tensor(sorted_mix_wav)
        sorted_mix_wav_len = torch.from_numpy(sorted_mix_wav_len)
        sorted_aim_wav = torch.tensor(sorted_aim_wav)

        if use_cuda:
            oracle_wav_endpoint = oracle_wav_endpoint.cuda().float()
            ref_wav = ref_wav.cuda().float()
            ref_wav_len = ref_wav_len.cuda().float()
            sorted_mix_wav = sorted_mix_wav.cuda().float()
            sorted_mix_wav_len = sorted_mix_wav_len.cuda()
            sorted_aim_wav = sorted_aim_wav.cuda().float()
        model.zero_grad()

        voiceP_outputs, predicted, oracle_endpoint, endpoint_0, endpoint_1, endpoint_2, endpoint_3 = model(
            sorted_mix_wav, ref_wav, ref_wav_len, oracle_wav_endpoint)

        if 1 and len(opt.gpus) > 1:
            voiceP_loss, num_sample, num_correct = model.module.voiceprint_loss(
                voiceP_outputs, voiceP_aim_spk_list)
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
            voiceP_loss, num_sample, num_correct = model.voiceprint_loss(
                voiceP_outputs, voiceP_aim_spk_list)
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


        loss = ss_loss + endpoint_loss
        loss.backward()


        optim.step()
        total_voiceP_loss += voiceP_loss.cpu().item()
        total_ss_loss += ss_loss.cpu().item()
        total_loss += loss.cpu().item()
        total_correct += num_correct.cpu().float().item()
        total_sample_num += num_sample.float()
        updates += 1


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
        logging('voiceP loss: %f' % voiceP_loss.cpu().item())
        logging('SS loss: %f' % ss_loss.cpu().item())
        logging('total loss: %f' % loss.cpu().item())
        logging('time: %f' % (time.time() - updates_start_time))
        writer.add_scalars(
            'scalar/endpoint_loss', {'endpoint_loss_0': endpoint_loss_0.cpu().item(), }, updates)
        writer.add_scalars(
            'scalar/endpoint_loss', {'endpoint_loss_1': endpoint_loss_1.cpu().item(), }, updates)
        writer.add_scalars(
            'scalar/endpoint_loss', {'endpoint_loss_2': endpoint_loss_2.cpu().item(), }, updates)
        writer.add_scalars(
            'scalar/endpoint_loss', {'endpoint_loss_3': endpoint_loss_3.cpu().item(), }, updates)
        writer.add_scalars(
            'scalar/endpoint_loss', {'endpoint_loss': endpoint_loss.cpu().item(), }, updates)

        writer.add_scalars(
            'scalar/loss', {'ss_loss': ss_loss.cpu().item()}, updates)

        if updates % config.voiceP_eval_interval == 0:
            logging(
                "time: %6.3f, epoch: %3d, updates: %8d, train loss: %6.3f, voiceP loss: %6.6f, ss loss: %6.6f, label acc: %6.6f\n"
                % (time.time() - voiceP_start_time, epoch, updates, loss / num_sample, total_voiceP_loss / config.voiceP_eval_interval,
                   total_ss_loss / config.voiceP_eval_interval, total_correct / total_sample_num))

            total_voiceP_loss, total_ss_loss = 0, 0
            voiceP_start_time = time.time()

        if updates > 100 and updates % config.eval_interval in range(1, 10):
            predicted = predicted[:, :-1, :]
            sorted_aim_wav = sorted_aim_wav[:, :-1, :]
            aim_spk_list = [[a[0]] for a in aim_spk_list]

            predicted /= torch.max(torch.abs(predicted),
                                   dim=2, keepdim=True)[0]

            try:
                sdr_aver_batch, sdri_aver_batch, sisnri_aver_batch = utils.bss_test.cal_using_wav(
                    config.batch_size, sorted_mix_wav, sorted_aim_wav, predicted)
                SDR_SUM = np.append(SDR_SUM, sdr_aver_batch)
                SDRi_SUM = np.append(SDRi_SUM, sdri_aver_batch)
                SISNRi_SUM = np.append(SISNRi_SUM, sisnri_aver_batch)
                logging('SDR_aver_now: %f' % SDR_SUM.mean())
                logging('SDRi_aver_now: %f' % SDRi_SUM.mean())
                logging('SISNRi_aver_now: %f' % SISNRi_SUM.mean())
                logging("time: %6.3f, epoch: %3d, updates: %8d, train loss: %6.5f, \
                    sdr_aver_batch: %4.2f, sdri_aver_batch: %4.2f, SDR_aver_now: %4.2f, SDRi_aver_now: %4.2f, SISNRi_aver_now: %4.2f\n"
                        % (time.time() - train_start_time, epoch, updates, total_loss / config.eval_interval,
                            sdr_aver_batch, sdri_aver_batch, SDR_SUM.mean(), SDRi_SUM.mean(), SISNRi_SUM.mean()))
                train_start_time = time.time()
            except AssertionError as wrong_info:  # ??
                logging('Errors in calculating the SDR: %s' % wrong_info)

        if updates > 100 and updates % config.eval_interval in range(9, 10):
            writer.add_scalars(
                'scalar/SDR', {'SDR_train': SDR_SUM.mean(), }, updates)

            writer.add_scalars(
                'scalar/SDR', {'SDR_train': SDR_SUM.mean(), }, updates)

        if 1 and updates % config.eval_interval == 0:
            logging("time: %6.3f, epoch: %3d, updates: %8d, train loss: %6.5f\n"
                    % (time.time() - eval_start_time, epoch, updates, total_loss / config.eval_interval))
            logging('evaluating after %d updates...\r' % updates)
            model.eval()
            score, loss_val = test(
                model, config, opt, writer, logging, updates)
            for metric in config.METRIC:
                scores[metric].append(score[metric])
                if metric == 'SDR' and score[metric] >= max(scores[metric]):
                    save_model(log_path + 'best_' + metric + '_checkpoint.pt')

            model.train()
            total_loss = 0
            eval_start_time = time.time()
            total_sample_num = 0
            total_correct = 0

        if updates % config.save_interval == 0:
            save_model(log_path + 'TDAAv3_{}.pt'.format(updates))


def save_model(path):
    global updates
    model_state_dict = model.module.state_dict() if len(
        opt.gpus) > 1 else model.state_dict()
    checkpoints = {
        'model': model_state_dict,
        'config': config,
        'updates': updates}
    torch.save(checkpoints, path)


def main():
    for i in range(1, config.epoch + 1):
        train(i)
    for metric in config.METRIC:
        logging("Best %s score: %.2f\n" % (metric, max(scores[metric])))


if __name__ == '__main__':
    main()
