from torch.utils.data import DistributedSampler, DataLoader, Dataset
from librosa.util import normalize
from scipy.io import wavfile
from hparams import hparams as hps, symbols
from hgtk.text import compose
from pathlib import Path
from typing import List
from tqdm import tqdm
import matplotlib.pyplot as plt
import unicodedata
import numpy as np
import torch
import soundfile as sf
import librosa
import glob
import os
import re

_mel_basis = None
_symbol_to_id = {s: i for i, s in enumerate(symbols.symbols)}
_id_to_symbol = {i: s for i, s in enumerate(symbols.symbols)}

# prerocessing functions
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

def decompose_hangul(sent: str) -> List[str]:
    res = []
    for hangul in sent:
        if re.match("[가-힣ㄱ-ㅎㅏ-ㅣ]", hangul) is None:
            res.append(hangul)
            continue
        if re.match("[ㄱ-ㅎ]", hangul) is not None:
            if hangul in symbols.special_ja:
                hangul = symbols.special_ja[hangul]
            else:
                middle = "ㅣ ㅇㅡ"
                if hangul == "ㄱ":
                    middle = "ㅣ ㅇㅕ"
                elif hangul == "ㄷ":
                    middle = "ㅣ ㄱㅡ"
                elif hangul == "ㅅ":
                    middle = "ㅣ ㅇㅗ"
                hangul = compose(hangul+middle+hangul+" ", compose_code=" ")
        elif re.match("[ㅏ-ㅣ]", hangul) is not None:
            hangul = compose("ㅇ"+hangul+" ", compose_code=" ")

        for c in hangul:
            # char_id = ord(c) - int('0xAC00', 16)
            # res.append(symbols.CHO[char_id // 28 // 21])
            # res.append(symbols.JOONG[char_id // 28 % 21])
            # if char_id % 28 != 0:
            #     res.append(symbols.JONG[char_id % 28])
            res += [jamo for jamo in unicodedata.normalize('NFKD', c)]
    return res

# text sequencing functions
def prep_text(text: str, conv_alpha: bool = False, conv_number: bool = False) -> List[str]:
    # lower -> change special words -> convert alphabet(optional) -> remove white space
    # -> remove non-eng/num/hangle/punc char -> convert number -> decompose hangle
    text = text.lower().strip()
    for r_exp, word in symbols.convert_symbols:
        text = re.sub(r_exp, word, text)

    if conv_alpha:
        text = re.sub("[a-z]", lambda x: symbols.alpha_pron[x.group(0)], text)
    if conv_number:
        text = re.sub("[0-9]+", convert_number, text)

    text = re.sub("\s+", " ", text)
    text = re.sub("[^0-9a-z가-힣ㄱ-ㅎㅏ-ㅣ ,.?!]", "", text)
    text = decompose_hangul(text)
    return text

def text_to_sequence(text: str) -> List[int]:
    text = prep_text(text, hps.convert_alpha, hps.convert_number)
    return [_symbol_to_id[s] for s in text if s in _symbol_to_id]

def sequence_to_text(sequence) -> str:
    return "".join([_id_to_symbol[s] for s in sequence if s in _id_to_symbol])

# dataloader
def prepare_dataloaders(data_dir: str, n_gpu: int) -> torch.utils.data.DataLoader:
    trainset = audio_dataset(data_dir)
    collate_fn = audio_collate(hps.n_frames_per_step)
    sampler = DistributedSampler(trainset) if n_gpu > 1 and torch.cuda.is_available() else None
    train_loader = DataLoader(trainset, num_workers=hps.n_workers, shuffle=n_gpu == 1,
                              batch_size=hps.batch_size, pin_memory=hps.pin_mem,
                              drop_last=True, collate_fn=collate_fn, sampler=sampler)
    return train_loader


# datasets
def _build_mel_basis():
    n_fft = (hps.num_freq - 1) * 2
    return librosa.filters.mel(hps.sample_rate, n_fft, n_mels=hps.num_mels, fmin=hps.fmin, fmax=hps.fmax)

def melspectrogram(y):
    # _stft(y)
    n_fft, hop_length, win_length = (hps.num_freq - 1) * 2, hps.frame_shift, hps.frame_length
    D = librosa.stft(y=y, n_fft=n_fft, hop_length=hop_length, win_length=win_length, pad_mode='reflect')

    # _amp_to_db(_linear_to_mel(np.abs(D)))
    global _mel_basis
    if _mel_basis is None:
        _mel_basis = _build_mel_basis()
    return np.log(np.maximum(1e-5, np.dot(_mel_basis, np.abs(D))))

def inv_melspectrogram(mel):
    # mel = _db_to_amp(mel)
    mel = np.exp(mel)

    # S = _mel_to_linear(mel)
    global _mel_basis
    if _mel_basis is None:
        _mel_basis = _build_mel_basis()
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


def get_text(text):
    return torch.IntTensor(text_to_sequence(text))

def get_mel(wav_path):
    sr, wav = wavfile.read(wav_path)
    assert sr == hps.sample_rate
    wav = normalize(wav.reshape(-1)/hps.MAX_WAV_VALUE)*0.95
    return torch.Tensor(melspectrogram(wav).astype(np.float32))

def get_mel_text_pair(text, wav_path):
    text = get_text(text)
    mel = get_mel(wav_path)
    return text, mel

def files_to_list(fdir):
    f_list = []
    for data_dir in os.listdir(fdir):
        if data_dir in hps.ignore_data_dir or \
                not os.path.exists(os.path.join(fdir, data_dir, 'transcript.txt')) or \
                not os.path.isdir(os.path.join(fdir, data_dir)):
            continue
        with open(os.path.join(fdir, data_dir, 'transcript.txt'), encoding='utf-8') as f:
            for line in tqdm(f, desc=f"loading data from {data_dir}"):
                parts = line.strip().split('|')
                wav_path = os.path.join(fdir, data_dir, parts[0])
                if hps.prep:
                    f_list.append(get_mel_text_pair(parts[1], wav_path))
                else:
                    f_list.append([parts[1], wav_path])

    assert f_list != []
    return f_list


class audio_dataset(Dataset):
    def __init__(self, fdir):
        self.f_list = files_to_list(fdir)

    def __getitem__(self, index):
        text, mel = self.f_list[index] if hps.prep else get_mel_text_pair(*self.f_list[index])
        return text, mel

    def __len__(self):
        return len(self.f_list)

class audio_collate:
    def __init__(self, n_frames_per_step):
        self.n_frames_per_step = n_frames_per_step

    def __call__(self, batch):
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
        sampling_rate = hps.sample_rate
        root_path = hps.default_data_path

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
