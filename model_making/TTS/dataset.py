import os
import re
import torch
import unicodedata
import random
import numpy as np
import matplotlib.pyplot as plt
from hgtk.text import compose
from typing import List

import glob
import librosa
import soundfile as sf
from tqdm import tqdm
from pathlib import Path
from scipy.io.wavfile import read

from model import TacotronSTFT
from hparams import symbols, hparams as hps
_symbol_to_id = {s: i for i, s in enumerate(symbols.symbols)}
_id_to_symbol = {i: s for i, s in enumerate(symbols.symbols)}

def split_train_val():
    train = ""
    val = ""
    for dir_path in ["../../data/TTS/trim_k_kwunT", "../../data/TTS/trim_kss"]:
        for line in open(os.path.join(dir_path, "transcript.txt"), "r+", encoding="utf-8").readlines():
            if random.random() > 0.2:
                train += dir_path + "/" + line
            else:
                val += dir_path + "/" + line
    open("../../data/TTS/train.txt", "w+", encoding="utf-8").write(train)
    open("../../data/TTS/val.txt", "w+", encoding="utf-8").write(val)

def get_number_of_digits(i: int) -> str:
    assert i != 0

    if i < 4:
        num_of_digits = symbols.number_of_digits[i - 1]
    elif i % 4 == 0:
        num_of_digits = symbols.number_of_digits[(i // 4) + 2]
    else:
        num_of_digits = symbols.number_of_digits[(i % 4) - 1] + symbols.number_of_digits[(i // 4) + 2]

    return num_of_digits

def convert_number(number: re.Match) -> str:
    number = number.group(0)
    number = number.lstrip("0")[::-1]
    if len(number) == 0:
        return symbols.digits[0]

    number_list = []
    for i, n in enumerate(number):
        n = int(n)
        if n == 0:
            continue

        if i == 0:
            digits = symbols.digits[n]
        elif n == 1:
            digits = get_number_of_digits(i)
        else:
            digits = symbols.digits[n] + get_number_of_digits(i)
        number_list.append(digits)

    return "".join(reversed(number_list))

def convert_cons_or_vowel(char: re.Match) -> str:
    char = char.group(0)
    if re.match("[ㄱ-ㅎ]", char) is not None:
        if char in symbols.special_ja:
            hangul = symbols.special_ja[char]
        else:
            middle = "ㅣ ㅇㅡ"
            if char == "ㄱ":
                middle = "ㅣ ㅇㅕ"
            elif char == "ㄷ":
                middle = "ㅣ ㄱㅡ"
            elif char == "ㅅ":
                middle = "ㅣ ㅇㅗ"
            hangul = compose(char + middle + char + " ", compose_code=" ")
    else:
        hangul = compose("ㅇ" + char + " ", compose_code=" ")
    return hangul


def text_to_sequence(text: str, conv_alpha: bool = True, conv_number: bool = True) -> List[int]:
    # text preprocessing
    text = text.lower()
    text = re.sub("[^0-9a-z가-힣ㄱ-ㅎㅏ-ㅣ ,.?!]", "", text)
    if conv_alpha:
        text = re.sub("[a-z]", lambda x: symbols.alpha_pron[x.group(0)], text)
    if conv_number:
        text = re.sub("[0-9]+", convert_number, text)
    text = re.sub("[ㄱ-ㅎㅏ-ㅣ]", convert_cons_or_vowel, text)
    # decompose hangul
    res = []
    for c in text:
        res += [jamo for jamo in unicodedata.normalize('NFKD', c)]
    # symbols to ids
    return [_symbol_to_id[s] for s in res if s in _symbol_to_id]

def sequence_to_text(sequence: List[int]) -> str:
    """Converts a sequence of IDs back to a string"""
    return "".join([_id_to_symbol[s] for s in sequence if s in _id_to_symbol])


class TextMelLoader(torch.utils.data.Dataset):
    """
        1) loads audio,text pairs
        2) normalizes text and converts them to sequences of one-hot vectors
        3) computes mel-spectrograms from audio files.
    """
    def __init__(self, audiopaths_and_text: str, hparams):
        self.audiopaths_and_text = self.load_filepaths_and_text(audiopaths_and_text)
        self.max_wav_value = hparams.max_wav_value
        self.sampling_rate = hparams.sampling_rate
        self.load_mel_from_disk = hparams.load_mel_from_disk
        self.ignore_dir = hparams.ignore_dir
        self.stft = TacotronSTFT(
            hparams.filter_length, hparams.hop_length, hparams.win_length,
            hparams.n_mel_channels, hparams.sampling_rate, hparams.mel_fmin,
            hparams.mel_fmax)
        random.seed(hparams.seed)
        random.shuffle(self.audiopaths_and_text)

    def load_filepaths_and_text(self, filename, split="|"):
        with open(filename, encoding='utf-8') as f:
            filepaths_and_text = [line.strip().split(split) for line in f
                                  if not any(ignore in line.strip().split(split)[0] for ignore in self.ignore_dir)]
        return filepaths_and_text

    def get_mel_text_pair(self, audiopath_and_text):
        # separate filename and text
        audiopath, text = audiopath_and_text[0], audiopath_and_text[2]
        text = self.get_text(text)
        mel = self.get_mel(audiopath)
        return text, mel

    def get_mel(self, filename):
        if not self.load_mel_from_disk:
            audio, sampling_rate = self.load_wav_to_torch(filename)
            if sampling_rate != self.stft.sampling_rate:
                raise ValueError("{} SR doesn't match target {} SR".format(
                    sampling_rate, self.stft.sampling_rate))
            audio_norm = audio / self.max_wav_value
            audio_norm = audio_norm.unsqueeze(0)
            audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
            melspec = self.stft.mel_spectrogram(audio_norm)
            melspec = torch.squeeze(melspec, 0)
        else:
            melspec = torch.from_numpy(np.load(filename))
            assert melspec.size(0) == self.stft.n_mel_channels, \
                ('Mel dimension mismatch: given {}, expected {}'.format(melspec.size(0), self.stft.n_mel_channels))
        return melspec

    def get_text(self, text):
        text_norm = torch.IntTensor(text_to_sequence(text))
        return text_norm

    def load_wav_to_torch(self, full_path):
        sampling_rate, data = read(full_path)
        return torch.FloatTensor(data.astype(np.float32)), sampling_rate

    def __getitem__(self, index):
        return self.get_mel_text_pair(self.audiopaths_and_text[index])

    def __len__(self):
        return len(self.audiopaths_and_text)

class TextMelCollate:
    """ Zero-pads model inputs and targets based on number of frames per step """
    def __init__(self, n_frames_per_step):
        self.n_frames_per_step = n_frames_per_step

    def __call__(self, batch):
        """ Collate's training batch from normalized text and mel-spectrogram
        PARAMS
        ------
        batch: [text_normalized, mel_normalized]
        """
        # Right zero-pad all one-hot text sequences to max input length
        input_lengths, ids_sorted_decreasing = torch.sort(torch.LongTensor([len(x[0]) for x in batch]), dim=0, descending=True)
        max_input_len = input_lengths[0]

        text_padded = torch.LongTensor(len(batch), max_input_len)
        text_padded.zero_()
        for i in range(len(ids_sorted_decreasing)):
            text = batch[ids_sorted_decreasing[i]][0]
            text_padded[i, :text.size(0)] = text

        # Right zero-pad mel-spec
        num_mels = batch[0][1].size(0)
        max_target_len = max([x[1].size(1) for x in batch])
        if max_target_len % self.n_frames_per_step != 0:
            max_target_len += self.n_frames_per_step - max_target_len % self.n_frames_per_step
            assert max_target_len % self.n_frames_per_step == 0

        # include mel padded and gate padded
        mel_padded = torch.FloatTensor(len(batch), num_mels, max_target_len)
        mel_padded.zero_()
        gate_padded = torch.FloatTensor(len(batch), max_target_len)
        gate_padded.zero_()
        output_lengths = torch.LongTensor(len(batch))
        for i in range(len(ids_sorted_decreasing)):
            mel = batch[ids_sorted_decreasing[i]][1]
            mel_padded[i, :, :mel.size(1)] = mel
            gate_padded[i, mel.size(1)-1:] = 1
            output_lengths[i] = mel.size(1)

        return text_padded, input_lengths, mel_padded, gate_padded, output_lengths

class audio_preprocesser:
    def __init__(self):
        self.run()

    def trim_audio(self, wav, top_db=10, pad_len=4000):
        # remove space
        non_silence_indices = librosa.effects.split(wav, top_db=top_db)
        start = non_silence_indices[0][0]
        end = non_silence_indices[-1][1]
        # cutting audio
        wav = wav[start:end]
        # add padding
        wav = np.hstack([np.zeros(pad_len), wav, np.zeros(pad_len)])
        return wav

    def plot_wav(self, wav, sr):
        plt.figure(1)

        plot_a = plt.subplot(211)
        plot_a.plot(wav)
        plot_a.set_xlabel('sample rate * time')
        plot_a.set_ylabel('energy')

        plot_b = plt.subplot(212)
        plot_b.specgram(wav, NFFT=1024, Fs=sr, noverlap=900)
        plot_b.set_xlabel('Time')
        plot_b.set_ylabel('Frequency')

        plt.show()

    def run(self):
        # remove smaller sound than specific decibel(depending personal setting)
        decibel = 10
        sampling_rate = hps.sampling_rate
        root_path = None

        for dir_name in os.listdir(root_path):
            save_path = os.path.join(root_path, f"trim_{dir_name}")
            os.makedirs(save_path, exist_ok=True)

            for sub_dir_name in os.listdir(os.path.join(root_path, dir_name)):
                if not os.path.isdir(os.path.join(root_path, dir_name, sub_dir_name)):
                    continue
                os.makedirs(os.path.join(root_path, dir_name, sub_dir_name), exist_ok=True)
                file_list = glob.glob(os.path.join(root_path, dir_name, sub_dir_name, "*.wav"))

                for file_path in tqdm(file_list, desc=f"{dir_name}/{sub_dir_name} files converting"):
                    wav, sr = librosa.load(file_path, sr=sampling_rate, mono=True)

                    trimed_wav = self.trim_audio(wav, top_db=decibel)

                    filename = Path(file_path).name
                    temp_save_path = os.path.join(save_path, sub_dir_name, filename)

                    sf.write(temp_save_path, trimed_wav, sampling_rate)
