import torch
import os
import time
import argparse
import numpy as np
import matplotlib.pylab as plt
from typing import Optional, Sequence

import simpleaudio
import librosa
from scipy.io import wavfile
from matplotlib import font_manager, rc

from model import Tacotron2
from glow import WaveGlow, Denoiser
from hparams import hparams as hps
from dataset import text_to_sequence
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
    def __init__(self, tacotron_check, waveglow_check):
        """
        Sound Synthesizer.
        Using Tacotron2 model and WaveGlow Vocoder from NVIDIA.

        :arg tacotron_check: checkpoint of Tacotron2 model path.
        :arg waveglow_check : checkpoint of WaveGlow model path.
        """
        self.text = ""
        self.outputMel = None
        self.n_mel_channels = 80
        self.sampling_rate = hps.sampling_rate

        model = Tacotron2()
        model = self.load_model(tacotron_check, model)

        # waveglow = WaveGlow(n_mel_channels=hps.num_mels, n_flows=12, n_group=8, n_early_every=4,
        #                     n_early_size=2, WN_config={"n_layers": 8, "n_channels": 256, "kernel_size": 3})
        # waveglow = self.load_model(waveglow_check, waveglow)

        if torch.cuda.is_available():
            model.cuda().eval()
            # waveglow.cuda().eval()
        else:
            model.eval()
            # waveglow.eval()

        self.tacotron = model
        # self.denoiser = Denoiser(waveglow)
        # self.waveglow = waveglow

    def synthesize(self, text, denoise=True, sigma=0.666):
        """
        make sound from input sound.
        if you want to synthesize phrase(not one sentence), using **synthesize_phrase** method.

        :param text: text for convert to sound.
        :param denoise: condition for process denoising.
        :param sigma: sigma for used in Waveglow inference. if increase this, (?) and decrease this, (?)

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
        # with torch.no_grad():
        #     audio = self.waveglow.infer(mel_outputs_postnet, sigma=sigma)
        # if denoise:
        #     audio_denoised = self.denoiser(audio, strength=0.01)[:, 0].cpu().numpy()
        #     audio = audio_denoised.reshape(-1)
        # else:
        #     audio = audio[0].data.cpu().numpy()
        audio = griffin_lim(self.to_arr(mel_outputs_postnet[0]))
        audio *= hps.MAX_WAV_VALUE
        audio = audio.astype(np.int16)

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
            wav_postnet = griffin_lim(self.to_arr(mel_outputs_postnet[0]))
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
    last_ckpt_path = "../../models/TTS/Tacotron2/ckpt/NVIDIA/checkpoint_160000"
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--ckpt_pth', type=str, default=last_ckpt_path, help='path to load Tacotron checkpoints')
    parser.add_argument('-i', '--img_pth', type=str, default='../../res/res_img.png', help='path to save images(png)')
    parser.add_argument('-w', '--wav_pth', type=str, default='../../res/res_wav.wav', help='path to save wavs(wav)')
    parser.add_argument('-n', '--npy_pth', type=str, default='../../res/res_npy.npy', help='path to save mels(npy)')
    parser.add_argument('-p', '--play_audio', type=bool, default=True, help='condition of playing generated audio.')
    parser.add_argument('-t', '--text', type=str, default='타코트론 모델 입니다.', help='text to synthesize')

    args = parser.parse_args()

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False

    syn = Synthesizer(args.ckpt_pth, "../../models/TTS/waveglow/waveglow_256channels_universal_v5.pt")
    syn_audio, sample_rate = syn.synthesize(args.text, denoise=False)

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
