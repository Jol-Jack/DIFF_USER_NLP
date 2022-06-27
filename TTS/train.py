import os
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt

import torch
import tensorboardX
import torch.distributed as dist

from dataset import prepare_dataloaders, griffin_lim, text_to_sequence
from model import Tacotron2, Tacotron2Loss
from hparams import hparams as hps
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

np.random.seed(hps.seed)
torch.manual_seed(hps.seed)
torch.cuda.manual_seed(hps.seed)

def save_checkpoint(ckpt_pth, model, optimizer, iteration):
    torch.save({'model': (model.module if hps.distributed else model).state_dict(),
                'optimizer': optimizer.state_dict(), 'iteration': iteration}, ckpt_pth)

def load_checkpoint(ckpt_pth, model, optimizer, device):
    ckpt_dict = torch.load(ckpt_pth, map_location=device)
    (model.module if hps.distributed else model).load_state_dict(ckpt_dict['model'])
    optimizer.load_state_dict(ckpt_dict['optimizer'])
    iteration = ckpt_dict['iteration']
    return model, optimizer, iteration

def to_arr(var) -> np.ndarray:
    return var.cpu().detach().numpy().astype(np.float32)

def infer(text, TTSmodel):
    sequence = text_to_sequence(text)
    sequence = torch.IntTensor(sequence)[None, :].to(hps.device).long()
    mel_outputs, mel_outputs_postnet, _, alignments = TTSmodel.inference(sequence)
    return mel_outputs, mel_outputs_postnet, alignments


class Tacotron2Logger(tensorboardX.SummaryWriter):
    def __init__(self, logdir):
        super(Tacotron2Logger, self).__init__(logdir, flush_secs=5)

    def log_training(self, items, grad_norm, learning_rate, iteration):
        self.add_scalar('loss.mel', items[0], iteration)
        self.add_scalar('loss.gate', items[1], iteration)
        self.add_scalar('grad.norm', grad_norm, iteration)
        self.add_scalar('learning.rate', learning_rate, iteration)

    def sample_train(self, outputs, iteration):
        mel_outputs = to_arr(outputs[0][0])
        mel_outputs_postnet = to_arr(outputs[1][0])
        alignments = to_arr(outputs[3][0]).T

        # plot alignment, mel and postnet output
        self.add_image('train.align', self.plot_alignment_to_numpy(alignments), iteration)
        self.add_image('train.mel', self.plot_spectrogram_to_numpy(mel_outputs), iteration)
        self.add_image('train.mel_post', self.plot_spectrogram_to_numpy(mel_outputs_postnet), iteration)

    def sample_infer(self, outputs, iteration):
        mel_outputs = to_arr(outputs[0][0])
        mel_outputs_postnet = to_arr(outputs[1][0])
        alignments = to_arr(outputs[2][0]).T

        # plot alignment, mel and postnet output
        self.add_image('infer.align', self.plot_alignment_to_numpy(alignments), iteration)
        self.add_image('infer.mel', self.plot_spectrogram_to_numpy(mel_outputs), iteration)
        self.add_image('infer.mel_post', self.plot_spectrogram_to_numpy(mel_outputs_postnet), iteration)

        # save audio
        wav = griffin_lim(mel_outputs)
        wav_postnet = griffin_lim(mel_outputs_postnet)
        self.add_audio('infer.wav', wav, iteration, hps.sample_rate)
        self.add_audio('infer.wav_post', wav_postnet, iteration, hps.sample_rate)

    def plot_alignment_to_numpy(self, alignment, info=None):
        fig, ax = plt.subplots(figsize=(6, 4))
        im = ax.imshow(alignment, aspect='auto', origin='lower', interpolation='none')
        fig.colorbar(im, ax=ax)
        xlabel = 'Decoder timestep'
        if info is not None:
            xlabel += '\n\n' + info
        plt.xlabel(xlabel)
        plt.ylabel('Encoder timestep')
        plt.tight_layout()

        fig.canvas.draw()
        data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        data = data.transpose(2, 0, 1)
        plt.close()
        return data

    def plot_spectrogram_to_numpy(self, spectrogram):
        fig, ax = plt.subplots(figsize=(12, 3))
        im = ax.imshow(spectrogram, aspect="auto", origin="lower", interpolation='none')
        plt.colorbar(im, ax=ax)
        plt.xlabel("Frames")
        plt.ylabel("Channels")
        plt.tight_layout()

        fig.canvas.draw()
        data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        data = data.transpose(2, 0, 1)
        plt.close()
        return data

def train(args):
    # setup env
    rank = 0
    local_rank = 1
    if hps.distributed:
        dist.init_process_group(backend='nccl', rank=local_rank, world_size=hps.n_workers)
    if local_rank:
        torch.cuda.set_device(local_rank)
    device = torch.device('cuda:{:d}'.format(local_rank))

    # build model
    model = Tacotron2()
    model.to(hps.device, non_blocking=hps.pin_mem)
    if hps.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
    optimizer = torch.optim.Adam(model.parameters(), lr=hps.lr, betas=hps.betas, eps=hps.eps, weight_decay=hps.weight_decay)
    criterion = Tacotron2Loss()

    # load checkpoint
    iteration = 1
    if hps.last_ckpt != '':
        model, optimizer, iteration = load_checkpoint(hps.last_ckpt, model, optimizer, device)
        iteration += 1

    # get scheduler
    if hps.sch:
        def scheduling(step) -> float:
            return hps.sch_step ** 0.5 * min((step + 1) * hps.sch_step ** -1.5, (step + 1) ** -0.5)

        if args.ckpt_pth != '':
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, scheduling, last_epoch=iteration)
        else:
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, scheduling)

    # make dataset
    train_loader = prepare_dataloaders(args.data_dir, hps.n_workers)

    if rank == 0:
        # get logger ready
        if args.log_dir != '':
            if not os.path.isdir(args.log_dir):
                os.makedirs(args.log_dir)
                os.chmod(args.log_dir, 0o775)
            logger = Tacotron2Logger(args.log_dir)

        # get ckpt_dir ready
        if args.ckpt_dir != '' and not os.path.isdir(args.ckpt_dir):
            os.makedirs(args.ckpt_dir)
            os.chmod(args.ckpt_dir, 0o775)

    model.train()
    # ================ MAIN TRAINING LOOP! ===================
    epoch = 0
    while iteration <= hps.max_iter:
        if hps.distributed:
            train_loader.sampler.set_epoch(epoch)
        for batch in train_loader:
            if iteration > hps.max_iter:
                break
            start = time.perf_counter()
            x, y = (model.module if hps.distributed else model).parse_batch(batch)
            y_pred = model(x)

            # loss
            loss, items = criterion(y_pred, y)

            # zero grad
            model.zero_grad()

            # backward, grad_norm, and update
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), hps.grad_clip_thresh)
            optimizer.step()
            if hps.sch:
                # noinspection PyUnboundLocalVariable
                scheduler.step()

            dur = time.perf_counter() - start
            if rank == 0:
                # info
                print('Iter: {} Mel Loss: {:.2e} Gate Loss: {:.2e} Grad Norm: {:.2e} {:.1f}s/it'.format(
                    iteration, items[0], items[1], grad_norm, dur))

                # log
                if args.log_dir != '' and (iteration % hps.iters_per_log == 0):
                    learning_rate = optimizer.param_groups[0]['lr']
                    # noinspection PyUnboundLocalVariable
                    logger.log_training(items, grad_norm, learning_rate, iteration)

                # sample
                if args.log_dir != '' and (iteration % hps.iters_per_sample == 0):
                    model.eval()
                    output = infer(hps.eg_text, model.module if hps.distributed else model)
                    model.train()
                    logger.sample_train(y_pred, iteration)
                    logger.sample_infer(output, iteration)

                # save ckpt
                if args.ckpt_dir != '' and (iteration % hps.iters_per_ckpt == 0):
                    ckpt_pth = os.path.join(args.ckpt_dir, 'ckpt_{}'.format(iteration))
                    save_checkpoint(ckpt_pth, model, optimizer, iteration)

            iteration += 1
        epoch += 1

    if rank == 0 and args.log_dir != '':
        logger.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_dir', type=str, default=hps.default_data_path, help='directory to load data')
    parser.add_argument('-l', '--log_dir', type=str, default=hps.default_log_path, help='directory to save tensorboard logs')
    parser.add_argument('-cd', '--ckpt_dir', type=str, default=hps.default_ckpt_path, help='directory to save checkpoints')
    parser.add_argument('-cp', '--ckpt_pth', type=str, default='', help='path to load checkpoints')

    train_args = parser.parse_args()

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    train(train_args)
