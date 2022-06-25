import torch
import os
import time
import argparse
import numpy as np
import matplotlib.pylab as plt
from typing import Optional, Sequence

import simpleaudio
from scipy.io import wavfile
from matplotlib import font_manager, rc
from speechbrain.pretrained import HIFIGAN

from model import Tacotron2
from hparams import hparams as hps
from dataset import text_to_sequence, inv_melspectrogram
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
font_path = "C:/Windows/Fonts/H2PORM.TTF"
font = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font)

class Synthesizer:
    def __init__(self, tacotron_check):
        """
        Sound Synthesizer.
        Using Tacotron2 model and WaveGlow Vocoder from NVIDIA.

        :arg tacotron_check: checkpoint of Tacotron2 model path.
        """
        self.text = ""
        self.outputMel = None
        self.n_mel_channels = 80
        self.sampling_rate = hps.sample_rate

        self.tacotron = Tacotron2()
        self.tacotron = self.load_model(tacotron_check, self.tacotron)
        self.hifi_gan = HIFIGAN.from_hparams(source="speechbrain/tts-hifigan-ljspeech", savedir="../../models/TTS/hifigan")
        if torch.cuda.is_available():
            self.tacotron.cuda().eval()
            self.hifi_gan.cuda().eval()
        else:
            self.tacotron.eval()
            self.hifi_gan.eval()

    def synthesize(self, text, use_griffin_lim: bool = False):
        """
        make sound from input sound.
        if you want to synthesize phrase(not one sentence), using **synthesize_phrase** method.

        :param text: text for convert to sound.
        :param use_griffin_lim

        :return: Sound : made sound, SamplingRate: sampling rate of made audio

        Example
        ------
        >>> synthesizer = Synthesizer("tacotron_path", "waveglow_path")
        >>> gen_audio, sr = synthesizer.synthesize("음성으로 변환할 텍스트")
        """
        self.text = text
        print("synthesize start")
        start = time.perf_counter()
        sequence = text_to_sequence(text)
        sequence = torch.IntTensor(sequence)[None, :].to(hps.device).long()
        mel_outputs, mel_outputs_postnet, _, alignments = self.tacotron.inference(sequence)
        self.outputMel = (mel_outputs, mel_outputs_postnet, alignments)

        if use_griffin_lim:
            # np.ndarray (T,)
            audio = inv_melspectrogram(self.to_arr(mel_outputs_postnet[0]))
            audio *= hps.MAX_WAV_VALUE
            audio = audio.astype(np.int16)
        else:
            # torch.Size([B, 1, T])
            audio = self.hifi_gan.decode_batch(mel_outputs_postnet).squeeze()
            audio *= hps.MAX_WAV_VALUE
            audio = self.to_arr(audio).astype(np.int16)

        print(f"synthesize text duration : {time.perf_counter()-start:.2f}sec.")
        return audio, self.sampling_rate

    def save_mel(self, pth):
        """
        save melspectrograms with npy.
        melspectrograms from synthesize method.
        have to processed after synthesize.
        :param pth: path for saving melspectrograms. has to end with '.npy'.

        Example
        -------
        >>> synthesizer = Synthesizer("tacotron_path", "waveglow_path")
        >>> gen_audio, sr = synthesizer.synthesize("음성으로 변환할 텍스트")
        >>> synthesizer.save_mel("result_mel.npy")
        """
        assert pth[-4:] == ".npy", "mel path has to end with '.npy'"
        assert self.outputMel, "save mel have to be processed after synthesize"
        mel_outputs, mel_outputs_postnet, _ = self.outputMel
        np.save(pth, self.to_arr(mel_outputs_postnet))

    def save_plot(self, pth):
        """
        save plots with image.
        plots consists of mel_output, mel_output_postnet, attention alignment.
        plots from synthesize method.
        have to processed after synthesize.
        :param pth: path for saving melspectrograms.

        Example
        -------
        >>> synthesizer = Synthesizer("tacotron_path", "waveglow_path")
        >>> gen_audio, sr = synthesizer.synthesize("음성으로 변환할 텍스트")
        >>> synthesizer.save_plot("result_plots.png")
        """
        assert self.outputMel, "save plot have to be processed after synthesize"
        self.plot_data([self.to_arr(plot[0]) for plot in self.outputMel], self.text)
        plt.savefig(pth)

    def save_wave(self, pth, outputAudio: Optional[Sequence[int]], use_griffin_lim=False):
        """
        save audio with wav form.

        case of use_griffin_lim is False,
        save wave with given audio. so have to input 'outputAudio'.
        outputAudio has to be audio data.

        case of use_griffin_lim is True,
        save wave with melspectrogram from synthsize method.
        so have to processed after synthesize and don't have to input 'outputAudio'.

        :param pth: path for saving audio.
        :param outputAudio: audio data for save with wav form.
        :param use_griffin_lim: condition of using griffin lim method.

        Example
        -------
        >>> synthesizer = Synthesizer("tacotron_path", "waveglow_path")
        >>> gen_audio, sr = synthesizer.synthesize("음성으로 변환할 텍스트")
        >>> synthesizer.save_wave("result_wav.wav", gen_audio)
        >>> synthesizer.save_wave("result_wav_using_griffin_lim.wav", use_griffin_lim=True)
        """
        assert pth[-4:] == ".wav", "wav path has to end with '.wav'"
        if use_griffin_lim:
            assert self.outputMel, "if you try to using griffin_lim method, you have to use synthesize method before."
            _, mel_outputs_postnet, _ = self.outputMel
            wav_postnet = inv_melspectrogram(self.to_arr(mel_outputs_postnet[0]))
            wav_postnet *= hps.MAX_WAV_VALUE
            wavfile.write(pth, self.sampling_rate, wav_postnet.astype(np.int16))
        else:
            assert outputAudio is not None, "for save_wave without griffin_lim, you have to input 'outputAudio'."
            wavfile.write(pth, self.sampling_rate, outputAudio)

    def load_model(self, ckpt_pth, model) -> torch.nn.Module:
        assert os.path.isfile(ckpt_pth)
        if torch.cuda.is_available():
            ckpt_dict = torch.load(ckpt_pth)
        else:
            ckpt_dict = torch.load(ckpt_pth, map_location=torch.device("cpu"))

        if isinstance(model, Tacotron2):
            model.load_state_dict(ckpt_dict['model'])
        else:
            model.load_state_dict(ckpt_dict['model'].state_dict())

        model = model.to(hps.device, non_blocking=True).eval()
        return model

    def plot_data(self, data, text, figsize=(16, 4)):
        data_order = ["melspectrogram", "melspectorgram_with_postnet", "attention_alignments"]
        fig, axes = plt.subplots(1, len(data), figsize=figsize)
        fig.suptitle(text)
        for i in range(len(data)):
            if data_order[i] == "attention_alignments":
                data[i] = data[i].T
            axes[i].imshow(data[i], aspect='auto', origin='lower')
            axes[i].set_title(data_order[i])
            if data_order[i] == "attention_alignments":
                axes[i].set_xlabel("Decoder TimeStep")
                axes[i].set_ylabel("Encoder TimeStep(Attention)")
            else:
                axes[i].set_xlabel("Time")
                axes[i].set_ylabel("Frequency")

    def to_arr(self, var) -> np.ndarray:
        return var.cpu().detach().numpy().astype(np.float32)


if __name__ == '__main__':
    last_ckpt_path = "../../models/TTS/Tacotron2/ckpt/BogiHsu/ckpt_300000"
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--ckpt_pth', type=str, default=last_ckpt_path, help='path to load Tacotron checkpoints')
    parser.add_argument('-i', '--img_pth', type=str, default='../../res/res_img.png', help='path to save images(png)')
    parser.add_argument('-w', '--wav_pth', type=str, default='../../res/res_wav.wav', help='path to save wavs(wav)')
    parser.add_argument('-n', '--npy_pth', type=str, default='../../res/res_npy.npy', help='path to save mels(npy)')
    parser.add_argument('-p', '--play_audio', type=bool, default=True, help='condition of playing generated audio.')
    parser.add_argument('-t', '--text', type=str, default='타코트론 모델이에요', help='text to synthesize')

    args = parser.parse_args()

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False

    syn = Synthesizer(args.ckpt_pth)
    syn_audio, sample_rate = syn.synthesize(args.text, use_griffin_lim=False)

    if args.img_pth != '':
        syn.save_plot(args.img_pth)
    if args.wav_pth != '':
        syn.save_wave(args.wav_pth, syn_audio)
    if args.npy_pth != '':
        syn.save_mel(args.npy_pth)
    if args.play_audio:
        wave_obj = simpleaudio.play_buffer(syn_audio, 1, 2, sample_rate)
        wave_obj.wait_done()

    print("generate ended")
