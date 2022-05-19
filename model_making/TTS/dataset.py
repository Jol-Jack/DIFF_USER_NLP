import re
import torch
from typing import List
from hparams import hparams as hps
from torch.utils.data import DistributedSampler, DataLoader

def text_to_sequence(text) -> List[int]:
    #  Converts a string of text to a sequence of IDs corresponding to the symbols in the text.
    #  The text can optionally have ARPAbet sequences enclosed in curly braces embedded in it.
    #  For example, "Turn left on {HH AW1 S S T AH0 N} Street.
    sequence = []
    while len(text):
        m = _curly_re.match(text)
        if not m:
            sequence += _symbols_to_sequence(_clean_text(text, cleaner_names))
            break
        sequence += _symbols_to_sequence(_clean_text(m.group(1), cleaner_names))
        sequence += _arpabet_to_sequence(m.group(2))
        text = m.group(3)

    return sequence

def sequence_to_text(sequence) -> str:
    result = ''
    for symbol_id in sequence:
        if symbol_id in _id_to_symbol:
            s = _id_to_symbol[symbol_id]
            # Enclose ARPAbet back in curly braces:
            if len(s) > 1 and s[0] == '@':
                s = '{%s}' % s[1:]
            result += s
    return result.replace('}{', ' ')


def prepare_dataloaders(data_dir: str, n_gpu: int) -> torch.utils.data.DataLoader:
    trainset = ljdataset(data_dir)
    collate_fn = ljcollate(hps.n_frames_per_step)
    sampler = DistributedSampler(trainset) if n_gpu > 1 else None
    train_loader = DataLoader(trainset, num_workers=hps.n_workers, shuffle=n_gpu == 1,
                              batch_size=hps.batch_size, pin_memory=hps.pin_mem,
                              drop_last=True, collate_fn=collate_fn, sampler=sampler)
    return train_loader
