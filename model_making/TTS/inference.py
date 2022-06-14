import torch
import argparse
import numpy as np
import matplotlib.pylab as plt
from matplotlib import font_manager, rc
from scipy.io import wavfile
from model import Tacotron2
from hparams import hparams as hps
from dataset import text_to_sequence, inv_melspectrogram
font_path = "C:/Windows/Fonts/H2PORM.TTF"
font = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font)

def to_arr(var) -> np.ndarray:
    return var.cpu().detach().numpy().astype(np.float32)

def load_model(ckpt_pth) -> torch.nn.Module:
    if torch.cuda.is_available():
        ckpt_dict = torch.load(ckpt_pth)
    else:
        ckpt_dict = torch.load(ckpt_pth, map_location=torch.device("cpu"))
    loaded_model = Tacotron2()
    loaded_model.load_state_dict(ckpt_dict['model'])
    loaded_model = loaded_model.to(hps.is_cuda, non_blocking=True).eval()
    return loaded_model

def infer(text, TTSmodel):
    sequence = text_to_sequence(text)
    sequence = torch.IntTensor(sequence)[None, :].to(hps.is_cuda).long()
    mel_outputs, mel_outputs_postnet, _, alignments = TTSmodel.inference(sequence)
    return mel_outputs, mel_outputs_postnet, alignments

def plot_data(data, text, figsize=(16, 4)):
    data_order = ["melspectrogram", "melspectorgram_with_postnet", "attention_alignments"]
    fig, axes = plt.subplots(1, len(data), figsize=figsize)
    fig.suptitle(text)
    for i in range(len(data)):
        axes[i].imshow(data[i], aspect='auto', origin='lower')
        axes[i].set_title(data_order[i])
        if data_order[i] == "attention_alignments":
            axes[i].set_xlabel("Decoder TimeStep")
            axes[i].set_ylabel("Encoder TimeStep(Attention)")
        else:
            axes[i].set_xlabel("Time")
            axes[i].set_ylabel("Frequency")


def save_plot(outputMel, pth, text):
    assert pth[-4:] == ".png", "plot path has to end with '.png'"
    mel_outputs, mel_outputs_postnet, alignments = outputMel
    plot_data((to_arr(mel_outputs[0]), to_arr(mel_outputs_postnet[0]), to_arr(alignments[0]).T), text)
    plt.savefig(pth)

def save_audio(outputMel, pth):
    assert pth[-4:] == ".wav", "wav path has to end with '.wav'"
    mel_outputs, mel_outputs_postnet, _ = outputMel
    wav_postnet = inv_melspectrogram(to_arr(mel_outputs_postnet[0]))

    wav_postnet *= hps.MAX_WAV_VALUE
    wavfile.write(pth, hps.sample_rate, wav_postnet.astype(np.int16))

def save_mel(outputMel, pth):
    assert pth[-4:] == ".npy", "mel path has to end with '.npy'"
    mel_outputs, mel_outputs_postnet, _ = outputMel
    np.save(pth, to_arr(mel_outputs_postnet))


if __name__ == '__main__':
    last_ckpt_path = "../../models/TTS/Tacotron2/ckpt/BogiHsu/ckpt_200000"
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--ckpt_pth', type=str, default=last_ckpt_path, help='path to load checkpoints')
    parser.add_argument('-i', '--img_pth', type=str, default='../../res/res_img.png', help='path to save images(png)')
    parser.add_argument('-w', '--wav_pth', type=str, default='../../res/res_wav.wav', help='path to save wavs(wav)')
    parser.add_argument('-n', '--npy_pth', type=str, default='../../res/res_npy.npy', help='path to save mels(npy)')
    parser.add_argument('-t', '--text', type=str, default='타코트론 모델 입니다.', help='text to synthesize')

    args = parser.parse_args()

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False

    model = load_model(args.ckpt_pth)
    output = infer(args.text, model)

    if args.img_pth != '':
        save_plot(output, args.img_pth, args.text)
    if args.wav_pth != '':
        save_audio(output, args.wav_pth)
    if args.npy_pth != '':
        save_mel(output, args.npy_pth)
    print("generate ended")
