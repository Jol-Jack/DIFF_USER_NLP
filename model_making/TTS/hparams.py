import torch

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
    n_workers = torch.cuda.device_count() - 1 if is_cuda == "cuda" else 2
    lr = 2e-3
    eps = 1e-5
    betas = (0.9, 0.999)
    weight_decay = 1e-6
    sch = True
    max_iter = 200e3
    batch_size = 16
    iters_per_log = 10
    iters_per_sample = 500
    iters_per_ckpt = 10000
    grad_clip_thresh = 1.0

    # params
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
