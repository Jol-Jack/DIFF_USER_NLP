import torch
import os

class symbols:
    _pad = '_'
    _punctuation = '!,.? '
    _alphabet = 'abcdefghijklmnopqrstuvwxyz'
    _numbers = '0123456789'
    CHO = [
        u'ᄀ', u'ᄁ', u'ᄂ', u'ᄃ', u'ᄄ', u'ᄅ', u'ᄆ', u'ᄇ', u'ᄈ', u'ᄉ',
        u'ᄊ', u'ᄋ', u'ᄌ', u'ᄍ', u'ᄎ', u'ᄏ', u'ᄐ', u'ᄑ', u'ᄒ'
    ]
    JOONG = [
        u'ᅡ', u'ᅢ', u'ᅣ', u'ᅤ', u'ᅥ', u'ᅦ', u'ᅧ', u'ᅨ', u'ᅩ', u'ᅪ',
        u'ᅫ', u'ᅬ', u'ᅭ', u'ᅮ', u'ᅯ', u'ᅰ', u'ᅱ', u'ᅲ', u'ᅳ', u'ᅴ', u'ᅵ',
    ]
    JONG = [
        u'', u'ᆨ', u'ᆩ', u'ᆪ', u'ᆫ', u'ᆬ', u'ᆭ', u'ᆮ', u'ᆯ', u'ᆰ',
        u'ᆱ', u'ᆲ', u'ᆳ', u'ᆴ', u'ᆵ', u'ᆶ', u'ᆷ', u'ᆸ', u'ᆹ', u'ᆺ',
        u'ᆻ', u'ᆼ', u'ᆽ', u'ᆾ', u'ᆿ', u'ᇀ', u'ᇁ', u'ᇂ'
    ]

    # special symbols
    special_ja = {"ㄲ": "쌍기역", "ㄸ": "쌍디귿", "ㅃ": "쌍비읍", "ㅆ": "쌍시옷", "ㅉ": "쌍지읒",
                  "ㄳ": "기역시옷", "ㄵ": "니은지읒", "ㄶ": "니은히읗", "ㄺ": "리을기역", "ㄻ": "리을미음",
                  "ㄼ": "리을비읍", "ㄽ": "리을시옷", "ㄾ": "리을티읕", "ㄿ": "리을피읖", "ㅀ": "리을히읗", "ㅄ": "비읍시옷"}
    alpha_pron = {"a": "에이", "b": "비", "c": "씨", "d": "디", "e": "이", "f": "에프", "g": "쥐",
                  "h": "에이치", "i": "아이", "j": "제이", "k": "케이", "l": "엘", "m": "엠", "n": "엔", "o": "오", "p": "피",
                  "q": "큐", "r": "알", "s": "에스", "t": "티", "u": "유", "v": "브이", "w": "더블유", "x": "엑스", "y": "와이",
                  "z": "지"}
    number_of_digits = ["십", "백", "천", "만", "억", "조", "경", "해"]
    digits = ["영", "일", "이", "삼", "사", "오", "육", "칠", "팔", "구"]

    # Export all symbols:
    symbols = [_pad] + CHO + JOONG + JONG[1:] + list(_alphabet) + list(_numbers) + list(_punctuation)

class hparams:
    ################################
    # Experiment Parameters        #
    ################################
    epochs = 500
    iters_per_checkpoint = 1000
    seed = 7777
    dynamic_loss_scaling = True
    distributed_run = False
    dist_backend = "nccl"
    dist_url = "tcp://localhost:54321"
    cudnn_enabled = torch.cuda.is_available()
    cudnn_benchmark = False
    ignore_layers = ['embedding.weight']

    ################################
    # Data Parameters             #
    ################################
    load_mel_from_disk = False
    ckp_for_transfer = None
    ignore_dir = 'trim_k_kwunT'
    training_files = '../../data/TTS/train.txt'
    validation_files = '../../data/TTS/val.txt'
    model_output_path = '../../models/TTS/ckpt'
    logging_dir = '../../models/TTS/log'

    ################################
    # Audio Parameters             #
    ################################
    max_wav_value = 32768.0
    sampling_rate = 22050
    filter_length = 1024
    hop_length = 256
    win_length = 1024
    n_mel_channels = 80
    mel_fmin = 0.0
    mel_fmax = 8000.0

    ################################
    # Model Parameters             #
    ################################
    n_symbols = len(symbols.symbols)
    symbols_embedding_dim = 512
    n_device = torch.cuda.device_count() if torch.cuda.is_available() else os.cpu_count() - 1

    # Encoder parameters
    encoder_kernel_size = 5
    encoder_n_convolutions = 3
    encoder_embedding_dim = 512

    # Decoder parameters
    n_frames_per_step = 1  # currently only 1 is supported
    decoder_rnn_dim = 1024
    prenet_dim = 256
    max_decoder_steps = 1000
    gate_threshold = 0.5
    p_attention_dropout = 0.1
    p_decoder_dropout = 0.1

    # Attention parameters
    attention_rnn_dim = 1024
    attention_dim = 128

    # Location Layer parameters
    attention_location_n_filters = 32
    attention_location_kernel_size = 31

    # Mel-post processing network parameters
    postnet_embedding_dim = 512
    postnet_kernel_size = 5
    postnet_n_convolutions = 5

    ################################
    # Optimization Hyperparameters #
    ################################
    use_saved_learning_rate = False
    learning_rate = 1e-3
    weight_decay = 1e-6
    grad_clip_thresh = 1.0
    batch_size = 64
    mask_padding = True  # set model's padded outputs to padded values
