import torch
import os
import json
import time
import argparse
import numpy as np
import matplotlib.pylab as plt

from scipy.io import wavfile
from matplotlib import font_manager, rc

from hparams import hparams as hps
from model import Tacotron2
from glow import WaveGlow, Denoiser
from dataset import text_to_sequence, inv_melspectrogram
font_path = "C:/Windows/Fonts/H2PORM.TTF"
font = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font)

def infer(text, TTSmodel):
    sequence = text_to_sequence(text)
    sequence = torch.IntTensor(sequence)[None, :].to(hps.device).long()
    mel_outputs, mel_outputs_postnet, _, alignments = TTSmodel.inference(sequence)
    return mel_outputs, mel_outputs_postnet, alignments

class Synthesizer:
    def __init__(self, tacotron_check, waveglow_check):
        """
        Sound Synthesizer.
        Using Tacotron2 model and WaveGlow Vocoder from NVIDIA.

        :arg tacotron_check: checkpoint of Tacotron2 model path.
        :arg waveglow_check : checkpoint of WaveGlow model path(dir).
        """
        self.n_mel_channels = 80
        self.sampling_rate = hps.sample_rate

        model = Tacotron2()
        model = self.load_model(tacotron_check, model)

        with open(waveglow_check+'/config.json') as f:
            data = f.read()
        config = json.loads(data)
        waveglow_config = config["waveglow_config"]

        waveglow = WaveGlow(**waveglow_config)
        waveglow = self.load_model(waveglow_check+"/waveglow_256channels_universal_v5.pt", waveglow)

        if torch.cuda.is_available():
            model.cuda().eval()
            waveglow.cuda().eval()
        else:
            model.eval()
            waveglow.eval()

        self.tacotron = model
        self.denoiser = Denoiser(waveglow)
        self.waveglow = waveglow

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

    def synthesize(self, text, denoise=True, return_tacotron_output=True, sigma=0.666):
        """
        make sound from input sound.
        if you want to synthesize phrase(not one sentence), using **synthesize_phrase** method.


        :param text: text for convert to sound.
        :param denoise: condition for process denoising.
        :param return_tacotron_output: condition for return outputs of tacotron2. consist of mel_output, mel_output_postnet, attention_alignment.
        :param sigma: sigma for used in Waveglow inference. if increase this, (?) and decrease this, (?)

        :return: Sound : made sound, SamplingRate: sampling rate of made audio, (Optional)tacotron_outputs(mel_output, mel_outputs_postnet, alignments)

        Example
        ------
        >>> synthesizer = Synthesizer("tacotron_path", "waveflow_path")
        >>> gen_audio, sr, tacoron_outputs = synthesizer.synthesize("음성으로 변환할 텍스트")
        """
        assert type(text) == str, "텍스트 하나만 지원합니다."
        print("synthesize start")
        start = time.perf_counter()
        sequence = text_to_sequence(text)
        sequence = torch.IntTensor(sequence)[None, :].to(hps.device).long()
        mel_outputs, mel_outputs_postnet, _, alignments = self.tacotron.inference(sequence)

        with torch.no_grad():
            audio = self.waveglow.infer(mel_outputs_postnet, sigma=sigma)
        if denoise:
            audio_denoised = self.denoiser(audio, strength=0.01)[:, 0].cpu().numpy()
            audio = audio_denoised.reshape(-1)
        else:
            audio = audio[0].data.cpu().numpy()

        print(f"synthesize text duration : {time.perf_counter()-start:.2f}sec.")
        if return_tacotron_output:
            return audio, self.sampling_rate, (mel_outputs, mel_outputs_postnet, alignments)
        else:
            return audio, self.sampling_rate

    def save_plot(self, outputMel, pth, text):
        assert pth[-4:] == ".png", "plot path has to end with '.png'"
        mel_outputs, mel_outputs_postnet, alignments = outputMel
        self.plot_data((self.to_arr(mel_outputs[0]), self.to_arr(mel_outputs_postnet[0]), self.to_arr(alignments[0]).T), text)
        plt.savefig(pth)

    def save_audio(self, outputAudio, outputMel, pth, using_tacoron_only=False):
        assert pth[-4:] == ".wav", "wav path has to end with '.wav'"
        if using_tacoron_only:
            mel_outputs, mel_outputs_postnet, _ = outputMel
            wav_postnet = inv_melspectrogram(self.to_arr(mel_outputs_postnet[0]))
            wav_postnet *= hps.MAX_WAV_VALUE
            wavfile.write(pth, self.sampling_rate, wav_postnet.astype(np.int16))
        else:
            wavfile.write(pth, self.sampling_rate, outputAudio)

    def save_mel(self, outputMel, pth):
        assert pth[-4:] == ".npy", "mel path has to end with '.npy'"
        mel_outputs, mel_outputs_postnet, _ = outputMel
        np.save(pth, self.to_arr(mel_outputs_postnet))

    def plot_data(self, data, text, figsize=(16, 4)):
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

    def to_arr(self, var) -> np.ndarray:
        return var.cpu().detach().numpy().astype(np.float32)


if __name__ == '__main__':
    last_ckpt_path = "../../models/TTS/Tacotron2/ckpt/BogiHsu/ckpt_200000"
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--ckpt_pth', type=str, default=last_ckpt_path, help='path to load Tacotron checkpoints')
    parser.add_argument('-i', '--img_pth', type=str, default='../../res/res_img.png', help='path to save images(png)')
    parser.add_argument('-w', '--wav_pth', type=str, default='../../res/res_wav.wav', help='path to save wavs(wav)')
    parser.add_argument('-n', '--npy_pth', type=str, default='../../res/res_npy.npy', help='path to save mels(npy)')
    parser.add_argument('-t', '--text', type=str, default='타코트론 모델 입니다.', help='text to synthesize')

    args = parser.parse_args()

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False

    syn = Synthesizer(args.ckpt_pth, "../../models/TTS/waveglow/")
    syn_audio, sample_rate, tacotron_outputs = syn.synthesize(args.text, denoise=False)

    if args.img_pth != '':
        syn.save_plot(tacotron_outputs, args.img_pth, args.text)
    if args.wav_pth != '':
        syn.save_audio(syn_audio, tacotron_outputs, args.wav_pth)
    if args.npy_pth != '':
        syn.save_mel(tacotron_outputs, args.npy_pth)
    print("generate ended")
