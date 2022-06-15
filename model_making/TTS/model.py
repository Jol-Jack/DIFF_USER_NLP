import torch
import numpy as np
from torch import nn
from math import sqrt
import librosa.util
from scipy.signal import get_window
from torch.autograd import Variable
from torch.nn import functional as F
from hparams import hparams as hps

def get_mask_from_lengths(lengths, pad=False) -> torch.Tensor:
    max_len = torch.max(lengths).item()
    if pad and max_len % hps.n_frames_per_step != 0:
        max_len += hps.n_frames_per_step - max_len % hps.n_frames_per_step
        assert max_len % hps.n_frames_per_step == 0
    ids = torch.arange(0, max_len, out=torch.LongTensor(max_len))
    ids = ids.to(hps.is_cuda)
    mask = (ids < lengths.unsqueeze(1))
    return mask

class LinearNorm(torch.nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, w_init_gain='linear'):
        super(LinearNorm, self).__init__()
        self.linear_layer = torch.nn.Linear(in_dim, out_dim, bias=bias)

        torch.nn.init.xavier_uniform_(self.linear_layer.weight, gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, x):
        return self.linear_layer(x)

class ConvNorm(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=None, dilation=1, bias=True, w_init_gain='linear'):
        super(ConvNorm, self).__init__()
        if padding is None:
            assert (kernel_size % 2 == 1)
            padding = int(dilation * (kernel_size - 1) / 2)

        self.conv = torch.nn.Conv1d(in_channels, out_channels, padding=padding, bias=bias,
                                    kernel_size=kernel_size, stride=stride, dilation=dilation)
        torch.nn.init.xavier_uniform_(self.conv.weight, gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, signal):
        conv_signal = self.conv(signal)
        return conv_signal

class LocationLayer(nn.Module):
    def __init__(self, attention_n_filters, attention_kernel_size, attention_dim):
        super(LocationLayer, self).__init__()
        padding = (attention_kernel_size - 1) // 2
        self.location_conv = ConvNorm(2, attention_n_filters, padding=padding, bias=False,
                                      kernel_size=attention_kernel_size, stride=1, dilation=1)
        self.location_dense = LinearNorm(attention_n_filters, attention_dim, bias=False, w_init_gain='tanh')

    def forward(self, attention_weights_cat):
        processed_attention = self.location_conv(attention_weights_cat)
        processed_attention = processed_attention.transpose(1, 2)
        processed_attention = self.location_dense(processed_attention)
        return processed_attention

class Attention(nn.Module):
    def __init__(self, attention_rnn_dim, embedding_dim, attention_dim,
                 attention_location_n_filters, attention_location_kernel_size):
        super(Attention, self).__init__()
        self.query_layer = LinearNorm(attention_rnn_dim, attention_dim, bias=False, w_init_gain='tanh')
        self.memory_layer = LinearNorm(embedding_dim, attention_dim, bias=False, w_init_gain='tanh')
        self.v = LinearNorm(attention_dim, 1, bias=False)
        self.location_layer = LocationLayer(attention_location_n_filters, attention_location_kernel_size, attention_dim)
        self.score_mask_value = -float('inf')

    def get_alignment_energies(self, query, processed_memory, attention_weights_cat):
        """
        PARAMS
        ------
        query: decoder output (batch, num_mels * n_frames_per_step)
        processed_memory: processed encoder outputs (B, T_in, attention_dim)
        attention_weights_cat: cumulative and prev. att weights (B, 2, max_time)

        RETURNS
        -------
        alignment (batch, max_time)
        """

        processed_query = self.query_layer(query.unsqueeze(1))
        processed_attention_weights = self.location_layer(attention_weights_cat)
        energies = self.v(torch.tanh(processed_query + processed_attention_weights + processed_memory))
        energies = energies.squeeze(-1)
        return energies

    def forward(self, attention_hidden_state, memory, processed_memory, attention_weights_cat, mask):
        """
        PARAMS
        ------
        attention_hidden_state: attention rnn last output
        memory: encoder outputs
        processed_memory: processed encoder outputs
        attention_weights_cat: previous and cummulative attention weights
        mask: binary mask for padded data
        """
        alignment = self.get_alignment_energies(attention_hidden_state, processed_memory, attention_weights_cat)

        if mask is not None:
            alignment.data.masked_fill_(mask, self.score_mask_value)

        attention_weights = F.softmax(alignment, dim=1)
        attention_context = torch.bmm(attention_weights.unsqueeze(1), memory)
        attention_context = attention_context.squeeze(1)
        return attention_context, attention_weights

class Tacotron2Loss(nn.Module):
    def __init__(self):
        super(Tacotron2Loss, self).__init__()
        self.loss = nn.MSELoss(reduction='none')

    def forward(self, model_outputs, targets):
        mel_out, mel_out_postnet, gate_out, _ = model_outputs
        gate_out = gate_out.view(-1, 1)

        mel_target, gate_target, output_lengths = targets
        mel_target.requires_grad = False
        gate_target.requires_grad = False
        output_lengths.requires_grad = False
        slice_arr = torch.arange(0, gate_target.size(1), hps.n_frames_per_step)
        gate_target = gate_target[:, slice_arr].view(-1, 1)
        mel_mask = ~get_mask_from_lengths(output_lengths.data, True)

        mel_loss = self.loss(mel_out, mel_target) + self.loss(mel_out_postnet, mel_target)
        mel_loss = mel_loss.sum(1).masked_fill_(mel_mask, 0.) / mel_loss.size(1)
        mel_loss = mel_loss.sum() / output_lengths.sum()

        gate_loss = nn.BCEWithLogitsLoss()(gate_out, gate_target)
        return mel_loss + gate_loss, (mel_loss.item(), gate_loss.item())


class Encoder(nn.Module):
    """Encoder module:
        - Three 1-d convolution banks
        - Bidirectional LSTM
    """

    def __init__(self):
        super(Encoder, self).__init__()

        convolutions = []
        for i in range(hps.encoder_n_convolutions):
            conv_layer = nn.Sequential(
                ConvNorm(hps.symbols_embedding_dim if i == 0 else hps.encoder_embedding_dim,
                         hps.encoder_embedding_dim,
                         kernel_size=hps.encoder_kernel_size, stride=1,
                         padding=(hps.encoder_kernel_size - 1) // 2,
                         dilation=1, w_init_gain='relu'),
                nn.BatchNorm1d(hps.encoder_embedding_dim))
            convolutions.append(conv_layer)
        self.convolutions = nn.ModuleList(convolutions)

        self.lstm = nn.LSTM(hps.encoder_embedding_dim, int(hps.encoder_embedding_dim / 2), 1, batch_first=True, bidirectional=True)

    def forward(self, x, input_lengths):
        for conv in self.convolutions:
            x = F.dropout(F.relu(conv(x)), 0.5, self.training)
        x = x.transpose(1, 2)

        # pytorch tensor are not reversible, hence the conversion
        input_lengths = input_lengths.cpu().numpy()
        x = nn.utils.rnn.pack_padded_sequence(x, input_lengths, batch_first=True)

        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(x)

        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        return outputs

    def inference(self, x):
        for conv in self.convolutions:
            x = F.dropout(F.relu(conv(x)), 0.5, self.training)
        x = x.transpose(1, 2)

        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(x)
        return outputs

class Prenet(nn.Module):
    def __init__(self, in_dim, sizes):
        super(Prenet, self).__init__()
        in_sizes = [in_dim] + sizes[:-1]
        self.layers = nn.ModuleList([LinearNorm(in_size, out_size, bias=False) for (in_size, out_size) in zip(in_sizes, sizes)])

    def forward(self, x):
        for linear in self.layers:
            x = F.dropout(F.relu(linear(x)), p=hps.dropout_rate, training=True)
        return x

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.attention_hidden = None
        self.attention_cell = None

        self.decoder_hidden = None
        self.decoder_cell = None

        self.attention_weights = None
        self.attention_weights_cum = None
        self.attention_context = None

        self.memory = None
        self.processed_memory = None
        self.mask = None

        self.prenet = Prenet(hps.num_mels * hps.n_frames_per_step, [hps.prenet_dim, hps.prenet_dim])

        self.attention_rnn = nn.LSTMCell(hps.prenet_dim + hps.encoder_embedding_dim, hps.attention_rnn_dim)

        self.attention_layer = Attention(
            hps.attention_rnn_dim, hps.encoder_embedding_dim,
            hps.attention_dim, hps.attention_location_n_filters,
            hps.attention_location_kernel_size)

        self.decoder_rnn = nn.LSTMCell(hps.attention_rnn_dim + hps.encoder_embedding_dim, hps.decoder_rnn_dim, True)

        self.linear_projection = LinearNorm(hps.decoder_rnn_dim + hps.encoder_embedding_dim, hps.num_mels * hps.n_frames_per_step)

        self.gate_layer = LinearNorm(hps.decoder_rnn_dim + hps.encoder_embedding_dim, 1, bias=True, w_init_gain='sigmoid')

    def get_go_frame(self, memory):
        """ Gets all zeros frames to use as first decoder input
        PARAMS
        ------
        memory: decoder outputs

        RETURNS
        -------
        decoder_input: all zeros frames
        """
        B = memory.size(0)
        decoder_input = Variable(memory.data.new(B, hps.num_mels * hps.n_frames_per_step).zero_())
        return decoder_input

    def initialize_decoder_states(self, memory, mask):
        """ Initializes attention rnn states, decoder rnn states, attention
        weights, attention cumulative weights, attention context, stores memory
        and stores processed memory
        PARAMS
        ------
        memory: Encoder outputs
        mask: Mask for padded data if training, expects None for inference
        """
        B = memory.size(0)
        MAX_TIME = memory.size(1)

        self.attention_hidden = Variable(memory.data.new(B, hps.attention_rnn_dim).zero_())
        self.attention_cell = Variable(memory.data.new(B, hps.attention_rnn_dim).zero_())

        self.decoder_hidden = Variable(memory.data.new(B, hps.decoder_rnn_dim).zero_())
        self.decoder_cell = Variable(memory.data.new(B, hps.decoder_rnn_dim).zero_())

        self.attention_weights = Variable(memory.data.new(B, MAX_TIME).zero_())
        self.attention_weights_cum = Variable(memory.data.new(B, MAX_TIME).zero_())
        self.attention_context = Variable(memory.data.new(B, hps.encoder_embedding_dim).zero_())

        self.memory = memory
        self.processed_memory = self.attention_layer.memory_layer(memory)
        self.mask = mask

    def parse_decoder_inputs(self, decoder_inputs):
        """ Prepares decoder inputs, i.e. mel outputs
        PARAMS
        ------
        decoder_inputs: inputs used for teacher-forced training, i.e. mel-specs

        RETURNS
        -------
        inputs: processed decoder inputs

        """
        # (B, num_mels, T_out) -> (B, T_out, num_mels)
        decoder_inputs = decoder_inputs.transpose(1, 2).contiguous()
        decoder_inputs = decoder_inputs.view(decoder_inputs.size(0), int(decoder_inputs.size(1) / hps.n_frames_per_step), -1)
        # (B, T_out, num_mels) -> (T_out, B, num_mels)
        decoder_inputs = decoder_inputs.transpose(0, 1)
        return decoder_inputs

    def parse_decoder_outputs(self, mel_outputs, gate_outputs, alignments):
        """ Prepares decoder outputs for output
        PARAMS
        ------
        mel_outputs:
        gate_outputs: gate output energies
        alignments:

        RETURNS
        -------
        mel_outputs:
        gate_outpust: gate output energies
        alignments:
        """
        # (T_out, B) -> (B, T_out)
        alignments = torch.stack(alignments).transpose(0, 1)
        # (T_out, B) -> (B, T_out)
        gate_outputs = torch.stack(gate_outputs).transpose(0, 1)
        gate_outputs = gate_outputs.contiguous()
        # (T_out, B, num_mels) -> (B, T_out, num_mels)
        mel_outputs = torch.stack(mel_outputs).transpose(0, 1).contiguous()
        # decouple frames per step
        mel_outputs = mel_outputs.view(mel_outputs.size(0), -1, hps.num_mels)
        # (B, T_out, num_mels) -> (B, num_mels, T_out)
        mel_outputs = mel_outputs.transpose(1, 2)
        return mel_outputs, gate_outputs, alignments

    def decode(self, decoder_input):
        """
        Decoder step using stored states, attention and memory
        PARAMS
        ------
        decoder_input: previous mel output

        RETURNS
        -------
        mel_output:
        gate_output: gate output energies
        attention_weights:
        """

        cell_input = torch.cat((decoder_input, self.attention_context), -1)
        self.attention_hidden, self.attention_cell = self.attention_rnn(cell_input, (self.attention_hidden, self.attention_cell))
        self.attention_hidden = F.dropout(self.attention_hidden, hps.p_attention_dropout, self.training)

        attention_weights_cat = torch.cat((self.attention_weights.unsqueeze(1), self.attention_weights_cum.unsqueeze(1)), dim=1)
        self.attention_context, self.attention_weights = self.attention_layer(
            self.attention_hidden, self.memory, self.processed_memory, attention_weights_cat, self.mask)

        self.attention_weights_cum += self.attention_weights
        decoder_input = torch.cat((self.attention_hidden, self.attention_context), -1)
        self.decoder_hidden, self.decoder_cell = self.decoder_rnn(decoder_input, (self.decoder_hidden, self.decoder_cell))
        self.decoder_hidden = F.dropout(self.decoder_hidden, hps.p_decoder_dropout, self.training)

        decoder_hidden_attention_context = torch.cat((self.decoder_hidden, self.attention_context), dim=1)
        decoder_output = self.linear_projection(decoder_hidden_attention_context)

        gate_prediction = self.gate_layer(decoder_hidden_attention_context)
        return decoder_output, gate_prediction, self.attention_weights

    def forward(self, memory, decoder_inputs, memory_lengths):
        """ Decoder forward pass for training
        PARAMS
        ------
        memory: Encoder outputs
        decoder_inputs: Decoder inputs for teacher forcing. i.e. mel-specs
        memory_lengths: Encoder output lengths for attention masking.

        RETURNS
        -------
        mel_outputs: mel outputs from the decoder
        gate_outputs: gate outputs from the decoder
        alignments: sequence of attention weights from the decoder
        """
        decoder_input = self.get_go_frame(memory).unsqueeze(0)
        decoder_inputs = self.parse_decoder_inputs(decoder_inputs)
        decoder_inputs = torch.cat((decoder_input, decoder_inputs), dim=0)
        decoder_inputs = self.prenet(decoder_inputs)

        self.initialize_decoder_states(memory, mask=~get_mask_from_lengths(memory_lengths))

        mel_outputs, gate_outputs, alignments = [], [], []
        while len(mel_outputs) < decoder_inputs.size(0) - 1:
            decoder_input = decoder_inputs[len(mel_outputs)]
            mel_output, gate_output, attention_weights = self.decode(decoder_input)
            mel_outputs += [mel_output.squeeze(1)]
            gate_outputs += [gate_output.squeeze()]
            alignments += [attention_weights]
        mel_outputs, gate_outputs, alignments = self.parse_decoder_outputs(mel_outputs, gate_outputs, alignments)
        return mel_outputs, gate_outputs, alignments

    def inference(self, memory):
        """ Decoder inference
        PARAMS
        ------
        memory: Encoder outputs

        RETURNS
        -------
        mel_outputs: mel outputs from the decoder
        gate_outputs: gate outputs from the decoder
        alignments: sequence of attention weights from the decoder
        """
        decoder_input = self.get_go_frame(memory)

        self.initialize_decoder_states(memory, mask=None)

        mel_outputs, gate_outputs, alignments = [], [], []
        while True:
            decoder_input = self.prenet(decoder_input)
            mel_output, gate_output, alignment = self.decode(decoder_input)

            mel_outputs += [mel_output.squeeze(1)]
            gate_outputs += [gate_output]
            alignments += [alignment]

            if torch.sigmoid(gate_output.data) > hps.gate_threshold:
                print('Terminated by gate.')
                break
            elif hps.n_frames_per_step * len(mel_outputs) / alignment.shape[1] >= hps.max_decoder_ratio:
                print('Warning: Reached max decoder steps.')
                break

            decoder_input = mel_output

        mel_outputs, gate_outputs, alignments = self.parse_decoder_outputs(mel_outputs, gate_outputs, alignments)
        return mel_outputs, gate_outputs, alignments

class Postnet(nn.Module):
    """
    Postnet
        - Five 1-d convolution with 512 channels and kernel size 5
    """

    def __init__(self):
        super(Postnet, self).__init__()
        self.convolutions = nn.ModuleList()

        self.convolutions.append(
            nn.Sequential(
                ConvNorm(hps.num_mels, hps.postnet_embedding_dim,
                         kernel_size=hps.postnet_kernel_size, stride=1,
                         padding=int((hps.postnet_kernel_size - 1) / 2),
                         dilation=1, w_init_gain='tanh'),
                nn.BatchNorm1d(hps.postnet_embedding_dim))
        )

        for i in range(1, hps.postnet_n_convolutions - 1):
            self.convolutions.append(
                nn.Sequential(
                    ConvNorm(hps.postnet_embedding_dim,
                             hps.postnet_embedding_dim,
                             kernel_size=hps.postnet_kernel_size, stride=1,
                             padding=int((hps.postnet_kernel_size - 1) / 2),
                             dilation=1, w_init_gain='tanh'),
                    nn.BatchNorm1d(hps.postnet_embedding_dim))
            )

        self.convolutions.append(
            nn.Sequential(
                ConvNorm(hps.postnet_embedding_dim, hps.num_mels,
                         kernel_size=hps.postnet_kernel_size, stride=1,
                         padding=int((hps.postnet_kernel_size - 1) / 2),
                         dilation=1, w_init_gain='linear'),
                nn.BatchNorm1d(hps.num_mels))
        )

    def forward(self, x):
        for i in range(len(self.convolutions) - 1):
            x = F.dropout(torch.tanh(self.convolutions[i](x)), 0.5, self.training)
        x = F.dropout(self.convolutions[-1](x), 0.5, self.training)
        return x

class Tacotron2(nn.Module):
    def __init__(self):
        super(Tacotron2, self).__init__()
        self.embedding = nn.Embedding(hps.n_symbols, hps.symbols_embedding_dim)
        std = sqrt(2.0 / (hps.n_symbols + hps.symbols_embedding_dim))
        val = sqrt(3.0) * std
        self.embedding.weight.data.uniform_(-val, val)
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.postnet = Postnet()

    def parse_batch(self, batch):
        text_padded, input_lengths, mel_padded, gate_padded, output_lengths = batch
        text_padded = text_padded.to(hps.is_cuda).long()
        input_lengths = input_lengths.to(hps.is_cuda).long()
        max_len = torch.max(input_lengths.data).item()
        mel_padded = mel_padded.to(hps.is_cuda).float()
        gate_padded = gate_padded.to(hps.is_cuda).float()
        output_lengths = output_lengths.to(hps.is_cuda).long()

        return (text_padded, input_lengths, mel_padded, max_len, output_lengths), (mel_padded, gate_padded, output_lengths)

    def parse_output(self, outputs, output_lengths=None):
        if output_lengths is not None:
            mask = ~get_mask_from_lengths(output_lengths, True)  # (B, T)
            mask = mask.expand(hps.num_mels, mask.size(0), mask.size(1))  # (80, B, T)
            mask = mask.permute(1, 0, 2)  # (B, 80, T)

            outputs[0].data.masked_fill_(mask, 0.0)  # (B, 80, T)
            outputs[1].data.masked_fill_(mask, 0.0)  # (B, 80, T)
            slice_arr = torch.arange(0, mask.size(2), hps.n_frames_per_step)
            outputs[2].data.masked_fill_(mask[:, 0, slice_arr], 1e3)  # gate energies (B, T//n_frames_per_step)
        return outputs

    def forward(self, inputs):
        text_inputs, text_lengths, mels, max_len, output_lengths = inputs
        text_lengths, output_lengths = text_lengths.data, output_lengths.data

        embedded_inputs = self.embedding(text_inputs).transpose(1, 2)

        encoder_outputs = self.encoder(embedded_inputs, text_lengths)

        mel_outputs, gate_outputs, alignments = self.decoder(encoder_outputs, mels, memory_lengths=text_lengths)

        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet
        return self.parse_output([mel_outputs, mel_outputs_postnet, gate_outputs, alignments], output_lengths)

    def inference(self, inputs):
        embedded_inputs = self.embedding(inputs).transpose(1, 2)
        encoder_outputs = self.encoder.inference(embedded_inputs)
        mel_outputs, gate_outputs, alignments = self.decoder.inference(encoder_outputs)

        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet

        outputs = self.parse_output([mel_outputs, mel_outputs_postnet, gate_outputs, alignments])
        return outputs

    def teacher_infer(self, inputs, mels):
        il, _ = torch.sort(torch.LongTensor([len(x) for x in inputs]), dim=0, descending=True)
        text_lengths = il.to(hps.is_cuda)

        embedded_inputs = self.embedding(inputs).transpose(1, 2)

        encoder_outputs = self.encoder(embedded_inputs, text_lengths)

        mel_outputs, gate_outputs, alignments = self.decoder(encoder_outputs, mels, memory_lengths=text_lengths)

        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet
        return self.parse_output([mel_outputs, mel_outputs_postnet, gate_outputs, alignments])


class STFT(torch.nn.Module):
    """adapted from Prem Seetharaman's https://github.com/pseeth/pytorch-stft"""
    def __init__(self, filter_length=800, hop_length=200, win_length=800, window='hann'):
        super(STFT, self).__init__()
        self.num_samples = None
        self.magnitude = self.phase = None
        self.filter_length = filter_length
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = window
        self.forward_transform = None
        scale = self.filter_length / self.hop_length
        fourier_basis = np.fft.fft(np.eye(self.filter_length))

        cutoff = int((self.filter_length / 2 + 1))
        fourier_basis = np.vstack([np.real(fourier_basis[:cutoff, :]), np.imag(fourier_basis[:cutoff, :])])

        forward_basis = torch.FloatTensor(fourier_basis[:, None, :])
        inverse_basis = torch.FloatTensor(np.linalg.pinv(scale * fourier_basis).T[:, None, :])

        if window is not None:
            assert(filter_length >= win_length)
            # get window and zero center pad it to filter_length
            fft_window = get_window(window, win_length, fftbins=True)
            fft_window = librosa.util.pad_center(fft_window, filter_length)
            fft_window = torch.from_numpy(fft_window).float()

            # window the bases
            forward_basis *= fft_window
            inverse_basis *= fft_window

        self.register_buffer('forward_basis', forward_basis.float())
        self.register_buffer('inverse_basis', inverse_basis.float())

    def transform(self, input_data):
        num_batches = input_data.size(0)
        num_samples = input_data.size(1)

        self.num_samples = num_samples

        # similar to librosa, reflect-pad the input
        input_data = input_data.view(num_batches, 1, num_samples)
        input_data = F.pad(
            input_data.unsqueeze(1),
            (int(self.filter_length / 2), int(self.filter_length / 2), 0, 0),
            mode='reflect')
        input_data = input_data.squeeze(1)

        forward_transform = F.conv1d(
            input_data, Variable(self.forward_basis, requires_grad=False), stride=self.hop_length, padding=0)

        cutoff = int((self.filter_length / 2) + 1)
        real_part = forward_transform[:, :cutoff, :]
        imag_part = forward_transform[:, cutoff:, :]

        magnitude = torch.sqrt(real_part**2 + imag_part**2)
        phase = torch.autograd.Variable(torch.atan2(imag_part.data, real_part.data))
        return magnitude, phase

    def window_sumsquare(self, window, n_frames, hop_length=200, win_length=800, n_fft=800, dtype=np.float32, norm=None):
        """
        # from librosa 0.6
        Compute the sum-square envelope of a window function at a given hop length.
        This is used to estimate modulation effects induced by windowing
        observations in short-time fourier transforms.
        Parameters
        ----------
        window : string, tuple, number, callable, or list-like
            Window specification, as in `get_window`
        n_frames : int > 0
            The number of analysis frames
        hop_length : int > 0
            The number of samples to advance between frames
        win_length : [optional]
            The length of the window function.  By default, this matches `n_fft`.
        n_fft : int > 0
            The length of each analysis frame.
        dtype : np.dtype
            The data type of the output
        norm : normalize
        Returns
        -------
        wss : np.ndarray, shape=`(n_fft + hop_length * (n_frames - 1))`
            The sum-squared envelope of the window function
        """
        if win_length is None:
            win_length = n_fft

        n = n_fft + hop_length * (n_frames - 1)
        x = np.zeros(n, dtype=dtype)

        # Compute the squared window at the desired length
        win_sq = get_window(window, win_length, fftbins=True)
        win_sq = librosa.util.normalize(win_sq, norm=norm) ** 2
        win_sq = librosa.util.pad_center(win_sq, n_fft)

        # Fill the envelope
        for i in range(n_frames):
            sample = i * hop_length
            x[sample:min(n, sample + n_fft)] += win_sq[:max(0, min(n_fft, n - sample))]
        return x

    def inverse(self, magnitude, phase):
        recombine_magnitude_phase = torch.cat([magnitude*torch.cos(phase), magnitude*torch.sin(phase)], dim=1)

        inverse_transform = F.conv_transpose1d(
            recombine_magnitude_phase,
            Variable(self.inverse_basis, requires_grad=False),
            stride=self.hop_length,
            padding=0)

        if self.window is not None:
            window_sum = self.window_sumsquare(
                self.window, magnitude.size(-1), hop_length=self.hop_length,
                win_length=self.win_length, n_fft=self.filter_length,
                dtype=np.float32)
            # remove modulation effects
            approx_nonzero_indices = torch.from_numpy(np.where(window_sum > librosa.util.tiny(window_sum))[0])
            window_sum = torch.autograd.Variable(torch.from_numpy(window_sum), requires_grad=False)
            window_sum = window_sum.cuda() if magnitude.is_cuda else window_sum
            inverse_transform[:, :, approx_nonzero_indices] /= window_sum[approx_nonzero_indices]

            # scale by hop ratio
            inverse_transform *= float(self.filter_length) / self.hop_length

        inverse_transform = inverse_transform[:, :, int(self.filter_length/2):]
        inverse_transform = inverse_transform[:, :, :-int(self.filter_length/2):]

        return inverse_transform

    def forward(self, input_data):
        self.magnitude, self.phase = self.transform(input_data)
        reconstruction = self.inverse(self.magnitude, self.phase)
        return reconstruction

class Denoiser(torch.nn.Module):
    """ Removes model bias from audio produced with waveglow """

    def __init__(self, waveglow, filter_length=1024, n_overlap=4, win_length=1024, mode='zeros'):
        super(Denoiser, self).__init__()
        self.stft = STFT(filter_length=filter_length, hop_length=int(filter_length/n_overlap), win_length=win_length).cuda()
        if mode == 'zeros':
            mel_input = torch.zeros((1, 80, 88), dtype=waveglow.upsample.weight.dtype, device=waveglow.upsample.weight.device)
        elif mode == 'normal':
            mel_input = torch.randn((1, 80, 88), dtype=waveglow.upsample.weight.dtype, device=waveglow.upsample.weight.device)
        else:
            raise Exception("Mode {} if not supported".format(mode))

        with torch.no_grad():
            bias_audio = waveglow.infer(mel_input, sigma=0.0).float()
            bias_spec, _ = self.stft.transform(bias_audio)

        self.register_buffer('bias_spec', bias_spec[:, :, 0][:, :, None])

    def forward(self, audio, strength=0.1):
        audio_spec, audio_angles = self.stft.transform(audio.cuda().float())
        audio_spec_denoised = audio_spec - self.bias_spec * strength
        audio_spec_denoised = torch.clamp(audio_spec_denoised, 0.0)
        audio_denoised = self.stft.inverse(audio_spec_denoised, audio_angles)
        return audio_denoised

class Invertible1x1Conv(torch.nn.Module):
    """
    The layer outputs both the convolution, and the log determinant
    of its weight matrix.  If reverse=True it does convolution with
    inverse
    """
    def __init__(self, c):
        super(Invertible1x1Conv, self).__init__()
        self.W_inverse = None
        self.conv = torch.nn.Conv1d(c, c, kernel_size=1, stride=1, padding=0, bias=False)

        # Sample a random orthonormal matrix to initialize weights
        W = torch.qr(torch.FloatTensor(c, c).normal_())[0]

        # Ensure determinant is 1.0 not -1.0
        if torch.det(W) < 0:
            W[:, 0] = -1*W[:, 0]
        W = W.view(c, c, 1)
        self.conv.weight.data = W

    def forward(self, z, reverse=False):
        # shape
        batch_size, group_size, n_of_groups = z.size()
        W = self.conv.weight.squeeze()

        if reverse:
            if not hasattr(self, 'W_inverse'):
                # Reverse computation
                W_inverse = W.float().inverse()
                W_inverse = Variable(W_inverse[..., None])
                if z.type() == 'torch.cuda.HalfTensor':
                    W_inverse = W_inverse.half()
                self.W_inverse = W_inverse
            z = F.conv1d(z, self.W_inverse, bias=None, stride=1, padding=0)
            return z
        else:
            # Forward computation
            log_det_W = batch_size * n_of_groups * torch.logdet(W)
            z = self.conv(z)
            return z, log_det_W

class WN(torch.nn.Module):
    """
    This is the WaveNet like layer for the affine coupling.  The primary difference
    from WaveNet is the convolutions need not be causal.  There is also no dilation
    size reset.  The dilation only doubles on each layer
    """
    def __init__(self, n_in_channels, n_mel_channels, n_layers, n_channels,
                 kernel_size):
        super(WN, self).__init__()
        assert(kernel_size % 2 == 1)
        assert(n_channels % 2 == 0)
        self.n_layers = n_layers
        self.n_channels = n_channels
        self.in_layers = torch.nn.ModuleList()
        self.res_skip_layers = torch.nn.ModuleList()

        start = torch.nn.Conv1d(n_in_channels, n_channels, 1)
        start = torch.nn.utils.weight_norm(start, name='weight')
        self.start = start

        # Initializing last layer to 0 makes the affine coupling layers
        # do nothing at first.  This helps with training stability
        end = torch.nn.Conv1d(n_channels, 2*n_in_channels, 1)
        end.weight.data.zero_()
        end.bias.data.zero_()
        self.end = end

        cond_layer = torch.nn.Conv1d(n_mel_channels, 2*n_channels*n_layers, 1)
        self.cond_layer = torch.nn.utils.weight_norm(cond_layer, name='weight')

        for i in range(n_layers):
            dilation = 2 ** i
            padding = int((kernel_size*dilation - dilation)/2)
            in_layer = torch.nn.Conv1d(n_channels, 2*n_channels, kernel_size, dilation=dilation, padding=padding)
            in_layer = torch.nn.utils.weight_norm(in_layer, name='weight')
            self.in_layers.append(in_layer)

            # last one is not necessary
            if i < n_layers - 1:
                res_skip_channels = 2*n_channels
            else:
                res_skip_channels = n_channels
            res_skip_layer = torch.nn.Conv1d(n_channels, res_skip_channels, 1)
            res_skip_layer = torch.nn.utils.weight_norm(res_skip_layer, name='weight')
            self.res_skip_layers.append(res_skip_layer)

    def forward(self, forward_input):
        audio, spect = forward_input
        audio = self.start(audio)
        output = torch.zeros_like(audio)
        n_channels_tensor = torch.IntTensor([self.n_channels])

        spect = self.cond_layer(spect)

        for i in range(self.n_layers):
            spect_offset = i*2*self.n_channels
            acts = self.fused_add_tanh_sigmoid_multiply(
                self.in_layers[i](audio), spect[:, spect_offset:spect_offset+2*self.n_channels, :], n_channels_tensor)

            res_skip_acts = self.res_skip_layers[i](acts)
            if i < self.n_layers - 1:
                audio = audio + res_skip_acts[:, :self.n_channels, :]
                output = output + res_skip_acts[:, self.n_channels:, :]
            else:
                output = output + res_skip_acts

        return self.end(output)

    @torch.jit.script
    def fused_add_tanh_sigmoid_multiply(self, input_a, input_b, n_channels):
        n_channels_int = n_channels[0]
        in_act = input_a + input_b
        t_act = torch.tanh(in_act[:, :n_channels_int, :])
        s_act = torch.sigmoid(in_act[:, n_channels_int:, :])
        acts = t_act * s_act
        return acts

class WaveGlow(torch.nn.Module):
    def __init__(self, n_mel_channels, n_flows, n_group, n_early_every, n_early_size, WN_config):
        super(WaveGlow, self).__init__()

        self.upsample = torch.nn.ConvTranspose1d(n_mel_channels, n_mel_channels, 1024, stride=256)
        assert(n_group % 2 == 0)
        self.n_flows = n_flows
        self.n_group = n_group
        self.n_early_every = n_early_every
        self.n_early_size = n_early_size
        self.WN = torch.nn.ModuleList()
        self.convinv = torch.nn.ModuleList()

        n_half = int(n_group/2)

        # Set up layers with the right sizes based on how many dimensions
        # have been output already
        n_remaining_channels = n_group
        for k in range(n_flows):
            if k % self.n_early_every == 0 and k > 0:
                n_half = n_half - int(self.n_early_size/2)
                n_remaining_channels = n_remaining_channels - self.n_early_size
            self.convinv.append(Invertible1x1Conv(n_remaining_channels))
            self.WN.append(WN(n_half, n_mel_channels*n_group, **WN_config))
        self.n_remaining_channels = n_remaining_channels  # Useful during inference

    def forward(self, forward_input):
        """
        forward_input[0] = mel_spectrogram:  batch x n_mel_channels x frames
        forward_input[1] = audio: batch x time
        """
        spect, audio = forward_input

        #  Upsample spectrogram to size of audio
        spect = self.upsample(spect)
        assert(spect.size(2) >= audio.size(1))
        if spect.size(2) > audio.size(1):
            spect = spect[:, :, :audio.size(1)]

        spect = spect.unfold(2, self.n_group, self.n_group).permute(0, 2, 1, 3)
        spect = spect.contiguous().view(spect.size(0), spect.size(1), -1).permute(0, 2, 1)

        audio = audio.unfold(1, self.n_group, self.n_group).permute(0, 2, 1)
        output_audio = []
        log_s_list = []
        log_det_W_list = []

        for k in range(self.n_flows):
            if k % self.n_early_every == 0 and k > 0:
                output_audio.append(audio[:, :self.n_early_size, :])
                audio = audio[:, self.n_early_size:, :]

            audio, log_det_W = self.convinv[k](audio)
            log_det_W_list.append(log_det_W)

            n_half = int(audio.size(1)/2)
            audio_0 = audio[:, :n_half, :]
            audio_1 = audio[:, n_half:, :]

            output = self.WN[k]((audio_0, spect))
            log_s = output[:, n_half:, :]
            b = output[:, :n_half, :]
            audio_1 = torch.exp(log_s)*audio_1 + b
            log_s_list.append(log_s)

            audio = torch.cat([audio_0, audio_1], 1)

        output_audio.append(audio)
        return torch.cat(output_audio, 1), log_s_list, log_det_W_list

    def infer(self, spect, sigma=1.0):
        spect = self.upsample(spect)
        # trim conv artifacts. maybe pad spec to kernel multiple
        time_cutoff = self.upsample.kernel_size[0] - self.upsample.stride[0]
        spect = spect[:, :, :-time_cutoff]

        spect = spect.unfold(2, self.n_group, self.n_group).permute(0, 2, 1, 3)
        spect = spect.contiguous().view(spect.size(0), spect.size(1), -1).permute(0, 2, 1)

        if spect.type() == 'torch.cuda.HalfTensor':
            audio = torch.cuda.HalfTensor(spect.size(0), self.n_remaining_channels, spect.size(2)).normal_()
        else:
            audio = torch.cuda.FloatTensor(spect.size(0), self.n_remaining_channels, spect.size(2)).normal_()

        audio = torch.autograd.Variable(sigma*audio)

        for k in reversed(range(self.n_flows)):
            n_half = int(audio.size(1)/2)
            audio_0 = audio[:, :n_half, :]
            audio_1 = audio[:, n_half:, :]

            output = self.WN[k]((audio_0, spect))

            s = output[:, n_half:, :]
            b = output[:, :n_half, :]
            audio_1 = (audio_1 - b)/torch.exp(s)
            audio = torch.cat([audio_0, audio_1], 1)

            audio = self.convinv[k](audio, reverse=True)

            if k % self.n_early_every == 0 and k > 0:
                if spect.type() == 'torch.cuda.HalfTensor':
                    z = torch.cuda.HalfTensor(spect.size(0), self.n_early_size, spect.size(2)).normal_()
                else:
                    z = torch.cuda.FloatTensor(spect.size(0), self.n_early_size, spect.size(2)).normal_()
                audio = torch.cat((sigma*z, audio), 1)

        audio = audio.permute(0, 2, 1).contiguous().view(audio.size(0), -1).data
        return audio
