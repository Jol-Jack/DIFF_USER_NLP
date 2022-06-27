import torch
import os
import time
import numpy as np
import matplotlib.pylab as plt
from typing import Sequence

import simpleaudio
import librosa
from scipy.io import wavfile
from matplotlib import font_manager, rc
from speechbrain.pretrained import HIFIGAN

from model import Tacotron2
from hparams import hparams as hps
from dataset import text_to_sequence
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
font_path = "C:/Windows/Fonts/H2PORM.TTF"
font = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font)

def griffin_lim(mel, n_iters=30):
    # mel = _db_to_amp(mel)
    mel = np.exp(mel)

    # S = _mel_to_linear(mel)
    _mel_basis = librosa.filters.mel(hps.sampling_rate, (hps.num_freq - 1) * 2, n_mels=hps.n_mel_channels, fmin=hps.mel_fmin, fmax=hps.mel_fmax)

    inv_mel_basis = np.linalg.pinv(_mel_basis)
    inverse = np.dot(inv_mel_basis, mel)
    S = np.maximum(1e-10, inverse)

    # _griffin_lim(S ** hps.power)
    n_fft, hop_length, win_length = (hps.num_freq - 1) * 2, hps.hop_length, hps.filter_length
    angles = np.exp(2j * np.pi * np.random.rand(*S.shape))
    S_complex = np.abs(S).astype(complex)
    y = librosa.istft(S_complex * angles, hop_length=hop_length, win_length=win_length)
    for i in range(n_iters):
        stft = librosa.stft(y=y, n_fft=n_fft, hop_length=hop_length, win_length=win_length, pad_mode='reflect')
        angles = np.exp(1j * np.angle(stft))
        y = librosa.istft(S_complex * angles, hop_length=hop_length, win_length=win_length)
    return np.clip(y, a_max=1, a_min=-1)

class Synthesizer:
    def __init__(self, tacotron_check, vocoder_dir):
        """
        Sound Synthesizer.
        Using Tacotron2 model and WaveGlow Vocoder from NVIDIA.

        :arg tacotron_check: path of Tacotron2 model checkpoint.
        :arg vocoder_dir: dir including vocoder files.
        """
        self.text = ""
        self.outputMel = None
        self.n_mel_channels = 80
        self.sampling_rate = hps.sampling_rate

        self.tacotron = Tacotron2()
        self.tacotron = self.load_model(tacotron_check, self.tacotron)
        self.hifi_gan = HIFIGAN.from_hparams(source="speechbrain/tts-hifigan-ljspeech", savedir=vocoder_dir)
        if torch.cuda.is_available():
            self.tacotron.cuda().eval()
            self.hifi_gan.cuda().eval()
        else:
            self.tacotron.eval()
            self.hifi_gan.eval()

    def synthesize(self, text, use_griffin_lim: bool = False):
        """
        synthesize audio from input text.

        :param text: text for synthesize.
        :param use_griffin_lim: condition of using griffin_lim vocoder. it cause low quality sound(including machine sound, etc).
        :return: audio(ndarray(Time, )), sampling_rate(int), duration(float, sec)
        """
        self.text = text
        print("synthesize start")
        start = time.perf_counter()
        sequence = text_to_sequence(text)
        sequence = torch.IntTensor(sequence)[None, :].to(hps.device).long()
        mel_outputs, mel_outputs_postnet, _, alignments = self.tacotron.inference(sequence)
        self.outputMel = (mel_outputs, mel_outputs_postnet, alignments)

        if use_griffin_lim:
            audio = griffin_lim(self.to_arr(mel_outputs_postnet[0]))
            audio *= hps.MAX_WAV_VALUE
            audio = audio.astype(np.int16)
        else:
            audio = self.hifi_gan.decode_batch(mel_outputs_postnet).squeeze()
            audio *= hps.MAX_WAV_VALUE
            audio = self.to_arr(audio).astype(np.int16)

        end = time.perf_counter()
        print(f"synthesize text duration : {end-start:.2f}sec.")
        return audio, self.sampling_rate, end-start

    def save_plot(self, pth):
        """
        save plots with image.
        plots consists of mel_output, mel_output_postnet, attention alignment.
        plots from synthesize method.
        have to processed after synthesize.
        :param pth: path for saving melspectrograms.
        """
        assert self.outputMel, "save plot have to be processed after synthesize"
        if not os.path.exists(os.path.dirname(pth)):
            os.mkdir(os.path.dirname(pth))

        self.plot_data([self.to_arr(plot[0]) for plot in self.outputMel], self.text)
        plt.savefig(pth)

    def save_wave(self, pth, outputAudio: Sequence[int]):
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
        """
        assert pth[-4:] == ".wav", "wav path has to end with '.wav'"
        if not os.path.exists(os.path.dirname(pth)):
            os.mkdir(os.path.dirname(pth))

        wavfile.write(pth, self.sampling_rate, outputAudio)

    def play_audio(self, p_audio):
        wave_obj = simpleaudio.play_buffer(p_audio, 1, 2, self.sampling_rate)
        wave_obj.wait_done()

    def load_model(self, ckpt_pth, model) -> torch.nn.Module:
        assert os.path.exists(ckpt_pth)
        if torch.cuda.is_available():
            ckpt_dict = torch.load(ckpt_pth)
        else:
            ckpt_dict = torch.load(ckpt_pth, map_location=torch.device("cpu"))

        if isinstance(model, Tacotron2):
            model.load_state_dict(ckpt_dict['state_dict'])
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
    synthesizer = Synthesizer(r"D:\workspace\Git_project\DIFF_USER_NLP\models\TTS\Tacotron2\checkpoint_260000", "../models/TTS/hifigan/")
    g_audio, _, _ = synthesizer.synthesize("타코트론 모델입니다.")
    synthesizer.save_plot("../res/res.png")
    synthesizer.save_wave("../res/res.wav", g_audio)
    synthesizer.play_audio(g_audio)
