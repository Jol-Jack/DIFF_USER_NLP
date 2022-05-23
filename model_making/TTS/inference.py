import torch
import argparse
import numpy as np
import matplotlib.pylab as plt
from dataset import text_to_sequence, inv_melspectrogram
from model import Tacotron2
from train import to_arr
from hparams import hparams as hps
from scipy.io import wavfile

def infer(text, TTSmodel):
    sequence = text_to_sequence(text)
    sequence = torch.IntTensor(sequence)[None, :].to(hps.is_cuda).long()
    mel_outputs, mel_outputs_postnet, _, alignments = TTSmodel.inference(sequence)
    return mel_outputs, mel_outputs_postnet, alignments

def load_model(ckpt_pth) -> torch.nn.Module:
    ckpt_dict = torch.load(ckpt_pth)
    loaded_model = Tacotron2()
    loaded_model.load_state_dict(ckpt_dict['model'])
    loaded_model = loaded_model.to(hps.is_cuda, non_blocking=True).eval()
    return loaded_model

def plot_data(data, figsize=(16, 4)):
    fig, axes = plt.subplots(1, len(data), figsize=figsize)
    for i in range(len(data)):
        axes[i].imshow(data[i], aspect='auto', origin='bottom')

def save_plot(outputMel, pth):
    mel_outputs, mel_outputs_postnet, alignments = outputMel
    plot_data((to_arr(mel_outputs[0]), to_arr(mel_outputs_postnet[0]), to_arr(alignments[0]).T))
    plt.savefig(pth + '.png')

def save_audio(outputMel, pth):
    mel_outputs, mel_outputs_postnet, _ = outputMel
    wav_postnet = inv_melspectrogram(to_arr(mel_outputs_postnet[0]))

    wav_postnet *= hps.MAX_WAV_VALUE
    wavfile.write(pth+".wav", hps.sample_rate, wav_postnet.astype(np.int16))

def save_mel(outputMel, pth):
    mel_outputs, mel_outputs_postnet, _ = outputMel
    np.save(pth + '.npy', to_arr(mel_outputs_postnet))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--ckpt_pth', type=str, default=hps.default_ckpt_path, required=True, help='path to load checkpoints')
    parser.add_argument('-i', '--img_pth', type=str, default='', help='path to save images')
    parser.add_argument('-w', '--wav_pth', type=str, default='', help='path to save wavs')
    parser.add_argument('-n', '--npy_pth', type=str, default='', help='path to save mels')
    parser.add_argument('-t', '--text', type=str, default='타코트론 모델 입니다.', help='text to synthesize')

    args = parser.parse_args()

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False

    model = load_model(args.ckpt_pth)
    output = infer(args.text, model)

    if args.img_pth != '':
        save_plot(output, args.img_pth)
    if args.wav_pth != '':
        save_audio(output, args.wav_pth)
    if args.npy_pth != '':
        save_mel(output, args.npy_pth)
