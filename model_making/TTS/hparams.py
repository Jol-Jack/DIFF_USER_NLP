import torch

class symbols:
    pad = '[PAD]'
    special = '-'
    punctuation = "!'(),.:;? "
    letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
    valid_symbols = [
        'AA', 'AA0', 'AA1', 'AA2', 'AE', 'AE0', 'AE1', 'AE2', 'AH', 'AH0', 'AH1', 'AH2',
        'AO', 'AO0', 'AO1', 'AO2', 'AW', 'AW0', 'AW1', 'AW2', 'AY', 'AY0', 'AY1', 'AY2',
        'B', 'CH', 'D', 'DH', 'EH', 'EH0', 'EH1', 'EH2', 'ER', 'ER0', 'ER1', 'ER2', 'EY',
        'EY0', 'EY1', 'EY2', 'F', 'G', 'HH', 'IH', 'IH0', 'IH1', 'IH2', 'IY', 'IY0', 'IY1',
        'IY2', 'JH', 'K', 'L', 'M', 'N', 'NG', 'OW', 'OW0', 'OW1', 'OW2', 'OY', 'OY0',
        'OY1', 'OY2', 'P', 'R', 'S', 'SH', 'T', 'TH', 'UH', 'UH0', 'UH1', 'UH2', 'UW',
        'UW0', 'UW1', 'UW2', 'V', 'W', 'Y', 'Z', 'ZH'
    ]

    # Prepend "@" to ARPAbet symbols to ensure uniqueness (some are the same as uppercase letters):
    arpabet = ['@' + s for s in valid_symbols]

    # Export all symbols:
    symbols = [pad] + [special] + list(punctuation) + list(letters) + arpabet

class hparams:
    seed = 7777

    # audio
    num_mels = 80
    num_freq = 513
    sample_rate = 22050
    frame_shift = 256
    frame_length = 1024
    fmin = 0
    fmax = 8000
    power = 1.5
    gl_iters = 30

    # train
    is_cuda = "cuda" if torch.cuda.is_available() else "cpu"
    pin_mem = True
    n_workers = torch.cuda.device_count() - 1 if is_cuda == "cuda" else 2
    lr = 2e-3
    eps = 1e-5
    betas = (0.9, 0.999)
    weight_decay = 1e-6
    sch = True
    sch_step = 4000
    max_iter = 200e3
    batch_size = 16
    iters_per_log = 10
    iters_per_sample = 500
    iters_per_ckpt = 10000
    grad_clip_thresh = 1.0
    eg_text = 'OMAK is a thinking process which considers things always positively.'

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
