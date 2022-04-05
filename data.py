from typing import Union, Dict
from pathlib import Path
from tape.tokenizers import TAPETokenizer
from tape.datasets import pad_sequences
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import torch


class AMPDataset(Dataset):
    def __init__(
        self,
        data_file: Union[str, Path, pd.DataFrame],
        task_label: Union[str, list] = "AMP",
        max_pep_len=180,
        tokenizer: Union[str, TAPETokenizer] = 'iupac',
    ):
        if isinstance(data_file, pd.DataFrame):
            data = data_file
        else:
            data = pd.read_csv(data_file)
        if isinstance(tokenizer, str):
            tokenizer = TAPETokenizer(vocab=tokenizer)
        self.tokenizer = tokenizer

        sequences = data['Sequence']
        sequences = sequences.apply(lambda x: x[:max_pep_len])
        if task_label == 'AMP':
            labels = data['Label']
        else:
            labels = data.loc[:, task_label].astype('float')  # for BCEloss

        self.sequences = sequences
        self.targets = labels.to_numpy()

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int):
        seq = self.sequences[index]
        token_ids = self.tokenizer.encode(seq)
        input_mask = np.ones_like(token_ids)
        item = {
            'input_ids': token_ids,
            'input_mask': input_mask,
            'target': self.targets[index]
        }
        return item

    def collate_fn(self, batch) -> Dict[str, torch.Tensor]:
        elem = batch[0]
        batch = {key: [d[key] for d in batch] for key in elem}

        input_ids = torch.from_numpy(pad_sequences(batch['input_ids'], 0))
        input_mask = torch.from_numpy(pad_sequences(batch['input_mask'], 0))
        targets = torch.tensor(batch['target'])

        item = {
            'input_ids': input_ids,
            'input_mask': input_mask,
            'targets': targets
        }
        return item
