import os
import time
import torch
import librosa
import argparse
import numpy as np
import tensorboardX
import matplotlib.pyplot as plt
from dataset import prepare_dataloaders
from hparams import hparams as hps
from model import Tacotron2, Tacotron2Loss, infer

np.random.seed(hps.seed)
torch.manual_seed(hps.seed)
torch.cuda.manual_seed(hps.seed)
_mel_basis = None

class Tacotron2Logger(tensorboardX.SummaryWriter):
    def __init__(self, logdir):
        super(Tacotron2Logger, self).__init__(logdir, flush_secs=5)

    def log_training(self, items, grad_norm, learning_rate, iteration):
        self.add_scalar('loss.mel', items[0], iteration)
        self.add_scalar('loss.gate', items[1], iteration)
        self.add_scalar('grad.norm', grad_norm, iteration)
        self.add_scalar('learning.rate', learning_rate, iteration)

    def sample_train(self, outputs, iteration):
        mel_outputs = self.to_arr(outputs[0][0])
        mel_outputs_postnet = self.to_arr(outputs[1][0])
        alignments = self.to_arr(outputs[3][0]).T

        # plot alignment, mel and postnet output
        self.add_image('train.align', self.plot_alignment_to_numpy(alignments), iteration)
        self.add_image('train.mel', self.plot_spectrogram_to_numpy(mel_outputs), iteration)
        self.add_image('train.mel_post', self.plot_spectrogram_to_numpy(mel_outputs_postnet), iteration)

    def sample_infer(self, outputs, iteration):
        mel_outputs = self.to_arr(outputs[0][0])
        mel_outputs_postnet = self.to_arr(outputs[1][0])
        alignments = self.to_arr(outputs[2][0]).T

        # plot alignment, mel and postnet output
        self.add_image('infer.align', self.plot_alignment_to_numpy(alignments), iteration)
        self.add_image('infer.mel', self.plot_spectrogram_to_numpy(mel_outputs), iteration)
        self.add_image('infer.mel_post', self.plot_spectrogram_to_numpy(mel_outputs_postnet), iteration)

        # save audio
        wav = self.inv_melspectrogram(mel_outputs)
        wav_postnet = self.inv_melspectrogram(mel_outputs_postnet)
        self.add_audio('infer.wav', wav, iteration, hps.sample_rate)
        self.add_audio('infer.wav_post', wav_postnet, iteration, hps.sample_rate)

    def to_arr(self, var) -> np.ndarray:
        return var.cpu().detach().numpy().astype(np.float32)

    def inv_melspectrogram(self, mel):
        # mel = _db_to_amp(mel)
        mel = np.exp(mel)

        # S = _mel_to_linear(mel)
        global _mel_basis
        if _mel_basis is None:
            n_fft = (hps.num_freq - 1) * 2
            _mel_basis = librosa.filters.mel(hps.sample_rate, n_fft, n_mels=hps.num_mels, fmin=hps.fmin, fmax=hps.fmax)
        inv_mel_basis = np.linalg.pinv(_mel_basis)
        inverse = np.dot(inv_mel_basis, mel)
        S = np.maximum(1e-10, inverse)

        # _griffin_lim(S ** hps.power)
        n_fft, hop_length, win_length = (hps.num_freq - 1) * 2, hps.frame_shift, hps.frame_length
        angles = np.exp(2j * np.pi * np.random.rand(*S.shape))
        S_complex = np.abs(S).astype(np.complex)
        y = librosa.istft(S_complex * angles, hop_length=hop_length, win_length=win_length)
        for i in range(hps.gl_iters):
            stft = librosa.stft(y=y, n_fft=n_fft, hop_length=hop_length, win_length=win_length, pad_mode='reflect')
            angles = np.exp(1j * np.angle(stft))
            y = librosa.istft(S_complex * angles, hop_length=hop_length, win_length=win_length)
        return np.clip(y, a_max=1, a_min=-1)

    def plot_alignment_to_numpy(self, alignment, info=None):
        fig, ax = plt.subplots(figsize=(6, 4))
        im = ax.imshow(alignment, aspect='auto', origin='lower',
                       interpolation='none')
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
        im = ax.imshow(spectrogram, aspect="auto", origin="lower",
                       interpolation='none')
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


def load_checkpoint(ckpt_pth, model, optimizer, device, n_gpu):
    ckpt_dict = torch.load(ckpt_pth, map_location=device)
    (model.module if n_gpu > 1 else model).load_state_dict(ckpt_dict['model'])
    optimizer.load_state_dict(ckpt_dict['optimizer'])
    iteration = ckpt_dict['iteration']
    return model, optimizer, iteration


def save_checkpoint(model, optimizer, iteration, ckpt_pth, n_gpu):
    torch.save({'model': (model.module if n_gpu > 1 else model).state_dict(),
                'optimizer': optimizer.state_dict(), 'iteration': iteration}, ckpt_pth)


def train(args):
    # setup env
    rank = local_rank = 0
    n_gpu = torch.cuda.device_count()
    if 'WORLD_SIZE' in os.environ:
        os.environ['OMP_NUM_THREADS'] = str(hps.n_workers)
        rank = int(os.environ['RANK'])
        local_rank = int(os.environ['LOCAL_RANK'])
        n_gpu = int(os.environ['WORLD_SIZE'])
        # noinspection PyUnresolvedReferences
        torch.distributed.init_process_group(backend='nccl', rank=local_rank, world_size=n_gpu)
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda:{:d}'.format(local_rank))

    # build model
    model = Tacotron2()
    model.to(hps.is_cuda, non_blocking=hps.pin_mem)
    if n_gpu > 1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
    optimizer = torch.optim.Adam(model.parameters(), lr=hps.lr, betas=hps.betas, eps=hps.eps, weight_decay=hps.weight_decay)
    criterion = Tacotron2Loss()

    # load checkpoint
    iteration = 1
    if args.ckpt_pth != '':
        model, optimizer, iteration = load_checkpoint(args.ckpt_pth, model, optimizer, device, n_gpu)
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
    train_loader = prepare_dataloaders(args.data_dir, n_gpu)

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
        if n_gpu > 1:
            train_loader.sampler.set_epoch(epoch)
        for batch in train_loader:
            if iteration > hps.max_iter:
                break
            start = time.perf_counter()
            x, y = (model.module if n_gpu > 1 else model).parse_batch(batch)
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
                    output = infer(hps.eg_text, model.module if n_gpu > 1 else model)
                    model.train()
                    logger.sample_train(y_pred, iteration)
                    logger.sample_infer(output, iteration)

                # save ckpt
                if args.ckpt_dir != '' and (iteration % hps.iters_per_ckpt == 0):
                    ckpt_pth = os.path.join(args.ckpt_dir, 'ckpt_{}'.format(iteration))
                    save_checkpoint(model, optimizer, iteration, ckpt_pth, n_gpu)

            iteration += 1
        epoch += 1

    if rank == 0 and args.log_dir != '':
        logger.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # path
    parser.add_argument('-d', '--data_dir', type=str, default='../data/TTS', help='directory to load data')
    parser.add_argument('-l', '--log_dir', type=str, default='../models/TTS/log', help='directory to save tensorboard logs')
    parser.add_argument('-cd', '--ckpt_dir', type=str, default='../models/TTS/ckpt', help='directory to save checkpoints')
    parser.add_argument('-cp', '--ckpt_pth', type=str, default='../models/TTS/ckpt', help='path to load checkpoints')

    train_args = parser.parse_args()

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    train(train_args)