import os
import random
from dataclasses import dataclass, field
import numpy as np 
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

PAD_TOKEN = 1024

def parse_data(file_path, is_wav_file = False):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split('\t', 1)
            utterance_id = parts[0]
            if is_wav_file: 
                values = parts[1]
            else: # values are list of tokens
                values = eval(parts[1])[0]
            data.append((utterance_id, values))
    return data

class BoundaryDataset(Dataset):
    def __init__(self, token_file_path, phone_label_file_path, syllable_label_file_path, word_label_file_path, max_seq_len=512):
        self.phone_labels = self.open_label_files(phone_label_file_path)
        self.syllable_labels = self.open_label_files(syllable_label_file_path)
        self.word_labels = self.open_label_files(word_label_file_path)
        self.tokens = parse_data(token_file_path)
        self.max_seq_len = max_seq_len 

    def open_label_files(self, label_file_path): 
        return np.load(label_file_path, allow_pickle=True)

    def __len__(self):
        return len(self.tokens)

    def sample_segment_crop_start_idx(self, segment_tensor): 
        if segment_tensor.size(0) > self.max_seq_len:
            start_idx = random.randint(0, segment_tensor.size(0) - self.max_seq_len)
            return start_idx
        return 0 

    def __getitem__(self, idx):
        uttid, tokens = self.tokens[idx]
        tokens_tensor = torch.tensor(tokens, dtype=torch.long)
        phone_labels_tensor = torch.tensor(self.phone_labels[uttid], dtype=torch.long)
        syllable_labels_tensor = torch.tensor(self.syllable_labels[uttid], dtype=torch.long)
        word_labels_tensor = torch.tensor(self.word_labels[uttid], dtype=torch.long)

        # Crop sequences if longer than max_seq_len
        start_idx = self.sample_segment_crop_start_idx(tokens_tensor)

        return (uttid, 
                tokens_tensor[start_idx:start_idx + self.max_seq_len], 
                phone_labels_tensor[start_idx:start_idx + self.max_seq_len],
                syllable_labels_tensor[start_idx:start_idx + self.max_seq_len],
                word_labels_tensor[start_idx:start_idx + self.max_seq_len])

def collate_fn(batch):
    # Unzip the batch to separate sequences and their labels
    uttids, tokens, phone_labels, syllable_labels, word_labels = zip(*batch)
    
    # Pad sequences to the maximum length in the batch
    # pad_sequence automatically pads to the maximum length of sequences in the batch
    tokens_padded = pad_sequence(tokens, batch_first=True, padding_value=PAD_TOKEN)  
    phone_labels_padded = pad_sequence(phone_labels, batch_first=True, padding_value=PAD_TOKEN)  
    syllable_labels_padded = pad_sequence(syllable_labels, batch_first=True, padding_value=PAD_TOKEN)
    word_labels_padded = pad_sequence(word_labels, batch_first=True, padding_value=PAD_TOKEN)

    # Assuming no specific positions need to be masked within the source sequence, 
    # and the entire sequence can attend to itself fully:
    src_mask = None
 
    # Create src_key_padding_mask
    src_key_padding_mask = (tokens_padded == PAD_TOKEN)
    tgt_key_padding_mask = (phone_labels_padded == PAD_TOKEN)
    assert (src_key_padding_mask == tgt_key_padding_mask).all()

    #print(tokens_padded.shape, phone_labels_padded.shape)
    return uttids, tokens_padded, phone_labels_padded, syllable_labels_padded, word_labels_padded, src_mask, src_key_padding_mask


@dataclass
class DatasetConfig:
    token_file_path: str = ''
    phone_label_file_path: str = ''
    syllable_label_file_path: str = ''
    word_label_file_path: str = ''
    batch_size: int = 32
    max_seq_length: int = 512
    shuffle: bool = True  # Shuffling usually needed for training
    num_workers: int = 2

def get_train_loader(config: DatasetConfig):
    dataset = BoundaryDataset(
        token_file_path=config.token_file_path,
        phone_label_file_path=config.phone_label_file_path,
        syllable_label_file_path=config.syllable_label_file_path,
        word_label_file_path=config.word_label_file_path,
        max_seq_len=config.max_seq_length
    )
    return DataLoader(dataset, batch_size=config.batch_size, shuffle=config.shuffle, num_workers=config.num_workers, pin_memory=True, collate_fn=collate_fn)

def get_eval_loader(config: DatasetConfig):
    dataset = BoundaryDataset(
        token_file_path=config.token_file_path,
        phone_label_file_path=config.phone_label_file_path,
        syllable_label_file_path=config.syllable_label_file_path,
        word_label_file_path=config.word_label_file_path,
        max_seq_len=config.max_seq_length
    )
    return DataLoader(dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers, pin_memory=True, collate_fn=collate_fn)

