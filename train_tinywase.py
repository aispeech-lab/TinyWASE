# coding=utf8
from eval_tinywase import test
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
from anybit import QuaOp
import shutil

# config
parser = argparse.ArgumentParser(description='train.py')
parser.add_argument('-config', default='config.yaml', type=str,
                    help="config file")
parser.add_argument('-gpus', default=[0,1,2,3,4,5,6,7], nargs='+', type=int,
                    help="Use CUDA on the listed devices.")
# parser.add_argument('-restore', default="TDAAv3_10_ak8.pt", type=str,
parser.add_argument('-restore', default=None, type=str,
                    help="restore checkpoint for student")
# parser.add_argument('-pretrained', default=None, type=str,
parser.add_argument('-pretrained', default="./log/log_3x8/TDAAv3_10.pt", type=str,
                    help="pretrained checkpoint for student")    
parser.add_argument('-restore_teacher', default="log/log_3x8/TDAAv3_10.pt", type=str,  
                    help="restore checkpoint for teacher")     
parser.add_argument('-seed', default=1234, type=int,
                    help="Random seed")
parser.add_argument('-memory', default=False, type=bool,
                    help="memory efficiency")
parser.add_argument('-score_fc', default='linear', type=str,
                    help="score function")
parser.add_argument('-bit', default=3, type=int,
                    help="bit for weight quantization")
parser.add_argument('-temperature', default=10, type=int, # pretrain 10
                    help="temperature for quantization")
parser.add_argument('-ak', default=8, type=int,
                    help="bit for activation quantization, 0 do not quantize activation")
parser.add_argument('-log', default='log_3x8_tiny_pshare', type=str,
                    help="log directory of student model")
parser.add_argument('-evaluate', type=bool, default=False,
                    help="evaluation only")
parser.add_argument('-sharing', default=1, type=int,
                    help="weight sharing")


opt = parser.parse_args()
config = utils.util.read_config(opt.config)
print('OPT', opt)

# logging module
if not os.path.exists(config.log):
    os.mkdir(config.log)
if opt.log == '':
    log_path = config.log + 'log_{}bit'.format(opt.bit) +  '/'
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


if not opt.sharing:
    from models.qwase import *
else:
    from models.qwase_pshare import *


class DistillFramework(object):
    def __init__(self, config, opt, logging, writer, return_features=True):
        # initialize
        torch.manual_seed(opt.seed)
        torch.backends.cudnn.deterministic = True

        # params and config
        self.config = config
        self.opt = opt
        self.logger = logging
        self.writer = writer
        self.return_features = return_features

        # cuda
        self.use_cuda = torch.cuda.is_available() and len(opt.gpus) > 0
        config.use_cuda = self.use_cuda
        if self.use_cuda:
            torch.cuda.set_device(opt.gpus[0])
            torch.cuda.manual_seed(opt.seed)
        print("use cuda:", self.use_cuda)

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
        self.QA_flag = True # 默认量化activation

        self.model = wase_q(config, fre_size, frame_num,
                        num_labels, self.use_cuda, opt.score_fc, QA_flag=self.QA_flag, ak=opt.ak)

        # teacher model
        self.teacher_model = models.wase(config, fre_size, frame_num,
                    num_labels, self.use_cuda, opt.score_fc, teacher=True)
        logging(repr(self.model) + "\n")

        self.start_epoch = 0

        # create alpha and beta for TCN module to quantize weight
        if not opt.sharing:
            count = 103
        else:
            count = 39
        self.alpha = []
        self.beta = []
        for i in range(count):
            self.alpha.append(Variable(torch.FloatTensor([0.0]).cuda(), requires_grad=True))
            self.beta.append(Variable(torch.FloatTensor([0.0]).cuda(), requires_grad=True))
        self.init_T = 0
        self.curr_T = 0

        # optimizer
        self.optim = utils.optims.Optim(
            config.optim, config.learning_rate, config.max_grad_norm)
        self.optim.set_parameters(self.model.parameters())
        self.optim_alpha = utils.optims.Optim(
            config.optim, config.learning_rate, config.max_grad_norm)
        self.optim_alpha.set_parameters(self.alpha)
        self.optim_beta = utils.optims.Optim(
            config.optim, config.learning_rate, config.max_grad_norm)
        self.optim_beta.set_parameters(self.beta)

        self.updates = 0

        # restore
        if opt.restore:
            restore_path = os.path.join(log_path, opt.restore)
            print('loading checkpoint...\n', opt.restore)
            checkpoints = torch.load(restore_path, map_location={'cuda:4': 'cuda:0'})
            self.model.load_state_dict(checkpoints['model'])
            self.alpha = checkpoints['alpha']
            self.beta = checkpoints['beta']
            self.init_T = checkpoints['T']
            self.updates = checkpoints['updates']
            print('Continue training')

        if opt.restore_teacher:
            print('loading checkpoint for teacher model...', opt.restore_teacher)
            teacher_checkpoint = torch.load(opt.restore_teacher)
            self.teacher_model.load_state_dict(teacher_checkpoint['model'])

        if opt.pretrained:
            print('loading checkpoint from pretrained model...', opt.pretrained)
            checkpoint = torch.load(opt.pretrained)
            self.model.load_state_dict(checkpoint['model'])

        # load the weights of voiceprint encoder and freeze the weights here
        voiceprint_checkpoint = {k:v for k, v in teacher_checkpoint['model'].items()
            if (('ref_encoder' in k) or ('voiceprint_encoder' in k) or ('linear.' in k))}
        print('voiceprint network keys', voiceprint_checkpoint.keys())
        self.model.load_state_dict(voiceprint_checkpoint, strict=False)
        print('Loading the voiceprint network from teacher model!')

        if self.use_cuda:
            self.model.cuda()
            self.teacher_model.cuda()

        if len(opt.gpus) > 1:
            self.model = nn.DataParallel(self.model, device_ids=opt.gpus, dim=0)
            self.teacher_model = nn.DataParallel(self.teacher_model, device_ids=opt.gpus, dim=0)

        # parameter number
        param_count = 0
        for param in self.model.parameters():
            param_count += param.view(-1).size()[0]
        logging('parameter number: %d\n' % param_count)

        self.scores = [[] for metric in config.METRIC]
        self.scores = collections.OrderedDict(zip(config.METRIC, self.scores))

        # quantization
        if opt.bit == 1:
            qw_values = [-1, 1]
            numpy_file = os.path.join(log_path, 'bias.npy')
        else:
            qw_values = list(range(-2**(opt.bit-1)+1, 2**(opt.bit-1)))
            n = len(qw_values) - 1
            numpy_file = os.path.join(log_path, 'bias.npy')
        if not os.path.exists(numpy_file):
            QW_biases = []
            initialize_biases = True
        else:
            initialize_biases = False
            QW_biases = np.load(numpy_file)
            print('QW_bias', QW_biases)
            # npy to list!!
            QW_biases = list(QW_biases)

        print('QW_values {}'.format(qw_values))
        if len(opt.gpus) > 1:
            self.qua_op = QuaOp([self.model.module.ss_model.TCN], QW_biases, qw_values, initialize_biases=initialize_biases)
        else:
            self.qua_op = QuaOp([self.model.ss_model.TCN], QW_biases, qw_values, initialize_biases=initialize_biases)

        if initialize_biases:
            print('Save and freeze bias for quantization!')
            numpy_file = os.path.join(log_path, 'bias.npy')
            np.save(numpy_file, self.qua_op.QW_biases)

        self.mse_loss = nn.MSELoss()


    def train_an_epoch(self, epoch, T=1):
        logging = self.logger
        config = self.config
        opt = self.opt
        writer = self.writer

        voiceP_start_time, train_start_time, eval_start_time = time.time(), time.time(), time.time()
        
        total_voiceP_loss, total_ss_loss, total_loss, total_sample_num, total_correct = 0, 0, 0, 0, 0
        SDR_SUM = np.array([])
        SDRi_SUM = np.array([])
        SISNRi_SUM = np.array([])

        first_batch = True

        self.teacher_model.eval()
        self.model.train()

        logging("Epoch %g begin..." % epoch)
        logging("Decaying learning rate to %g" %
                self.optim.optimizer.param_groups[0]['lr'])

        train_data_gen = prepare_data('once', 'train')

        while True:
            if epoch == 1 and first_batch and opt.restore is None: 
                init = True
            else:
                init = False
            first_batch = False

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

            if self.use_cuda:
                oracle_wav_endpoint = oracle_wav_endpoint.cuda().float()
                ref_wav = ref_wav.cuda().float()
                ref_wav_len = ref_wav_len.cuda().float()
                sorted_mix_wav = sorted_mix_wav.cuda().float()
                sorted_mix_wav_len = sorted_mix_wav_len.cuda()
                sorted_aim_wav = sorted_aim_wav.cuda().float()
            self.model.zero_grad()

            # teacher model
            with torch.no_grad():
                if not self.return_features:
                    voiceP_outputs_teacher, predicted_teacher, oracle_endpoint_teacher, _, _, _, _ = self.teacher_model(
                        sorted_mix_wav, ref_wav, ref_wav_len, oracle_wav_endpoint, self.return_features)
                else:
                    voiceP_outputs_teacher, predicted_teacher, oracle_endpoint_teacher, _, _, _, _, [f1_t, f2_t, f3_t] = self.teacher_model(
                        sorted_mix_wav, ref_wav, ref_wav_len, oracle_wav_endpoint, self.return_features)
            # quantization
            self.qua_op.quantization(T, self.alpha, self.beta, init=init)
            if not self.return_features:
                voiceP_outputs, predicted, oracle_endpoint, endpoint_0, endpoint_1, endpoint_2, endpoint_3 = self.model(
                    sorted_mix_wav, ref_wav, ref_wav_len, oracle_wav_endpoint, self.return_features)
            else:
                voiceP_outputs, predicted, oracle_endpoint, endpoint_0, endpoint_1, endpoint_2, endpoint_3, [f1_s, f2_s, f3_s] = self.model(
                    sorted_mix_wav, ref_wav, ref_wav_len, oracle_wav_endpoint, self.return_features)
            if 1 and len(opt.gpus) > 1:
                voiceP_loss, num_sample, num_correct = self.model.module.voiceprint_loss(
                    voiceP_outputs, voiceP_aim_spk_list)
                ss_loss = self.model.module.separation_tas_loss(
                    sorted_aim_wav, predicted, sorted_mix_wav_len)
                endpoint_loss_0, accuracy_0, precision_0, recall_0, f1_0 = self.model.module.endpoint_loss(
                    endpoint_0, oracle_endpoint)
                endpoint_loss_1, accuracy_1, precision_1, recall_1, f1_1 = self.model.module.endpoint_loss(
                    endpoint_1, oracle_endpoint)
                endpoint_loss_2, accuracy_2, precision_2, recall_2, f1_2 = self.model.module.endpoint_loss(
                    endpoint_2, oracle_endpoint)
                endpoint_loss_3, accuracy_3, precision_3, recall_3, f1_3 = self.model.module.endpoint_loss(
                    endpoint_3, oracle_endpoint)
            else:
                voiceP_loss, num_sample, num_correct = self.model.voiceprint_loss(
                    voiceP_outputs, voiceP_aim_spk_list)
                ss_loss = self.model.separation_tas_loss(
                    sorted_aim_wav, predicted, sorted_mix_wav_len)
                endpoint_loss_0, accuracy_0, precision_0, recall_0, f1_0 = self.model.endpoint_loss(
                    endpoint_0, oracle_endpoint)
                endpoint_loss_1, accuracy_1, precision_1, recall_1, f1_1 = self.model.endpoint_loss(
                    endpoint_1, oracle_endpoint)
                endpoint_loss_2, accuracy_2, precision_2, recall_2, f1_2 = self.model.endpoint_loss(
                    endpoint_2, oracle_endpoint)
                endpoint_loss_3, accuracy_3, precision_3, recall_3, f1_3 = self.model.endpoint_loss(
                    endpoint_3, oracle_endpoint)
            endpoint_loss = torch.mean(torch.stack(
                (endpoint_loss_0, endpoint_loss_1, endpoint_loss_2, endpoint_loss_3), 0))

            loss = ss_loss + endpoint_loss

            # distillation
            distill_losses = []
            
            distill_losses.append(self.model.module.separation_tas_loss(predicted_teacher, predicted, sorted_mix_wav_len))
            distill_loss = 0.2 * distill_losses[0]
            loss += distill_loss

            loss.backward()
            
            self.qua_op.restore_params()
            alpha_grad, beta_grad = self.qua_op.updateQuaGradWeight(T, self.alpha, self.beta, init=init)
            for idx in range(len(self.alpha)):
                self.alpha[idx].grad = Variable(torch.FloatTensor([alpha_grad[idx]]).cuda())
                self.beta[idx].grad = Variable(torch.FloatTensor([beta_grad[idx]]).cuda())

            self.optim.step()
            self.optim_alpha.step()
            self.optim_beta.step()
            total_voiceP_loss += voiceP_loss.cpu().item()
            total_ss_loss += ss_loss.cpu().item()
            total_loss += loss.cpu().item()
            total_correct += num_correct.cpu().float().item()
            total_sample_num += num_sample.float()
            self.updates += 1
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
            logging('Distillation loss: %f' % distill_loss.cpu().item())                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          

            for idx, dloss in enumerate(distill_losses):
                logging('Distillation loss {}: {}'.format(idx, dloss.cpu().item()))

            logging('total loss: %f' % loss.cpu().item())
            logging('time: %f' % (time.time() - updates_start_time))
            writer.add_scalars(
                'scalar/endpoint_loss', {'endpoint_loss_0': endpoint_loss_0.cpu().item(), }, self.updates)
            writer.add_scalars(
                'scalar/endpoint_loss', {'endpoint_loss_1': endpoint_loss_1.cpu().item(), }, self.updates)
            writer.add_scalars(
                'scalar/endpoint_loss', {'endpoint_loss_2': endpoint_loss_2.cpu().item(), }, self.updates)
            writer.add_scalars(
                'scalar/endpoint_loss', {'endpoint_loss_3': endpoint_loss_3.cpu().item(), }, self.updates)
            writer.add_scalars(
                'scalar/endpoint_loss', {'endpoint_loss': endpoint_loss.cpu().item(), }, self.updates)

            writer.add_scalars('scalar/distillation_loss', {'distillation_loss': distill_loss.cpu().item()}, self.updates)
            writer.add_scalars(
                'scalar/loss', {'ss_loss': ss_loss.cpu().item()}, self.updates)

            if self.updates % config.voiceP_eval_interval == 0:
                logging(
                    "time: %6.3f, epoch: %3d, updates: %8d, voiceP loss: %6.6f, ss loss: %6.6f, label acc: %6.6f\n"
                    % (time.time() - voiceP_start_time, epoch, self.updates, total_voiceP_loss / config.voiceP_eval_interval,
                    total_ss_loss / config.voiceP_eval_interval, total_correct / total_sample_num))

                total_voiceP_loss, total_ss_loss = 0, 0
                voiceP_start_time = time.time()

            if self.updates > 100 and self.updates % config.eval_interval in range(1, 10):
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
                            % (time.time() - train_start_time, epoch, self.updates, total_loss / config.eval_interval,
                                sdr_aver_batch, sdri_aver_batch, SDR_SUM.mean(), SDRi_SUM.mean(), SISNRi_SUM.mean()))
                    train_start_time = time.time()
                except AssertionError as wrong_info:  # ??
                    logging('Errors in calculating the SDR: %s' % wrong_info)

            if self.updates > 100 and self.updates % config.eval_interval in range(9, 10):
                writer.add_scalars(
                    'scalar/SDR', {'SDR_train': SDR_SUM.mean(), }, self.updates)
                writer.add_scalars(
                    'scalar/SDR', {'SDR_train': SDR_SUM.mean(), }, self.updates)

            if 1 and self.updates % config.eval_interval == 0:
                logging("time: %6.3f, epoch: %3d, updates: %8d, train loss: %6.5f\n"
                        % (time.time() - eval_start_time, epoch, self.updates, total_loss / config.eval_interval))
                logging('evaluating after %d updates...\r' % self.updates)
                self.model.eval()

                # quantization
                print('alpha: ', self.alpha)
                print('beta: ', self.beta)
                self.qua_op.quantization(T, self.alpha, self.beta, init=False, train_phase=False)
                score, _ = test(
                    self.model, config, opt, writer, logging, self.updates, mode='valid', bit=opt.bit)
                for metric in config.METRIC:
                    self.scores[metric].append(score[metric])
                    if metric == 'SDR' and score[metric] >= max(self.scores[metric]):
                        self.save_model(log_path + 'best_' + metric + '_checkpoint_ak{}.pt'.format(opt.ak))
                self.qua_op.restore_params()

                if self.updates % config.save_interval == 0:
                    self.save_model(log_path + 'TDAAv3_{}_ak{}.pt'.format(self.updates, opt.ak))

                self.model.train()

    def save_model(self, fpath):
        model_state_dict = self.model.module.state_dict() if len(
            self.opt.gpus) > 1 else self.model.state_dict()
        checkpoints = {
            'model': model_state_dict,
            'config': self.config,
            'updates': self.updates,
            'QW_bias': self.qua_op.QW_biases,
            'QW_values': self.qua_op.QW_values,
            'bit': self.opt.bit,
            'alpha': self.alpha,
            'beta': self.beta,
            'QA_flag': self.QA_flag,
            'ak': self.opt.ak,
            'T': self.curr_T
            }
        torch.save(checkpoints, fpath)

    def train(self):
        start_epoch = 1
        for i in range(1, self.config.epoch + 1):
            print('alpha is {}'.format(self.alpha))
            print('beta is {}'.format(self.beta))
            self.curr_T = self.init_T + i * self.opt.temperature
            print('temperature is {}'.format(self.curr_T))
            self.train_an_epoch(i, self.curr_T)
        for metric in config.METRIC:
            self.logger("Best %s score: %.2f\n" % (metric, max(self.scores[metric])))


if __name__ == '__main__':
    distiller = DistillFramework(config, opt, logging, writer)
    distiller.train()
