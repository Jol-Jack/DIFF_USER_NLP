import os
import torch

class symbols:
    pad = '[PAD]'
    eos = '</s>'
    punctuation = "!,.? "
    alphabet = 'abcdefghijklmnopqrstuvwxyz'
    numbers = '0123456789'

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
    convert_symbols = [("(주)", "주식회사"), ("-([0-9]+)", r"마이너스\1"), ("%", "퍼센트")]
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
    symbols = [pad] + [eos] + CHO + JOONG + JONG[1:] + list(alphabet) + list(numbers) + list(punctuation)

class hparams:
    seed = 7777

    # audio
    MAX_WAV_VALUE = 32768.0
    num_mels = 80
    num_freq = 513
    fmin = 0
    fmax = 8000
    frame_shift = 256
    frame_length = 1024
    sample_rate = 22050
    power = 1.5
    gl_iters = 30

    # train
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_workers = torch.cuda.device_count() if torch.cuda.is_available() else 2
    # distributed = torch.cuda.is_available() and n_workers > 1
    distributed = False
    # if torch.cuda.is_available() and n_workers > 1:
    #     os.environ["WORLD_SIZE"] = str(n_workers)
    #     os.environ["RANK"] = "0"
    #     os.environ["LOCAL_RANK"] = "0"
    convert_alpha = True
    convert_number = True
    pin_mem = True
    prep = True
    lr = 2e-3
    eps = 1e-5
    betas = (0.9, 0.999)
    weight_decay = 1e-6
    dropout_rate = 0.4
    sch = True
    sch_step = 4000
    max_iter = 300e3
    batch_size = 16
    iters_per_log = 10
    iters_per_sample = 500
    iters_per_ckpt = 10000
    grad_clip_thresh = 1.0
    eg_text = '타코트론 모델의 성능 확인을 위한 예시 텍스트 입니다.'

    # path
    default_data_path = "../data/TTS"
    default_ckpt_path = "./models/Tacotron2/ckpt/BogiHsu"
    default_log_path = "./models/Tacotron2/log"
    last_ckpt = f"{default_ckpt_path}/ckpt_{max(int(ckpt.split('_')[1]) for ckpt in os.listdir(default_ckpt_path))}"\
        if os.path.exists(default_ckpt_path) and os.listdir(default_ckpt_path) else ""
    ignore_data_dir = ["trim_k_kwunT"]

    # params
    # model
    n_symbols = len(symbols.symbols)
    symbols_embedding_dim = 512
    # Encoder parameters
    encoder_kernel_size = 5
    encoder_n_convolutions = 3
    encoder_embedding_dim = 512
    # Decoder parameters
    n_frames_per_step = 3
    decoder_rnn_dim = 1024
    prenet_dim = 256
    max_decoder_ratio = 10
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
    postnet_kernel_size = 5
    postnet_n_convolutions = 5
    postnet_embedding_dim = 512
