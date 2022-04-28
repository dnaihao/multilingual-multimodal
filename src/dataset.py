import torch
import transformers
import multiprocessing

import pytorch_lightning as pl
from datasets import load_dataset
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from functools import partial
from overrides import overrides

from Multilingual_CLIP.src.multilingual_clip import MultilingualClip, AVAILABLE_MODELS


def collate_mm(examples, tokenizer, ln1, ln2):
    ln1s, ln2s, ln1s_atts, ln2s_atts = [], [], [], []
    for example in examples:
        ln1_ex, ln2_ex = example["translation"][ln1], example["translation"][ln2]
        encoded_ln1 = tokenizer(ln1_ex, return_tensors='pt')
        encoded_ln2 = tokenizer(ln2_ex, return_tensors='pt')
        ln1s.append(encoded_ln1["input_ids"]); ln2s.append(encoded_ln2["input_ids"])
        ln1s_atts.append(encoded_ln1["attention_mask"]); ln2s_atts.append(encoded_ln2["attention_mask"])
    return {
        ln1: torch.cat(ln1s, dim=0),
        ln2: torch.cat(ln2s, dim=0),
        f"{ln1}_attention_mask": torch.cat(ln1s_atts, dim=0),
        f"{ln2}_attention_mask": torch.cat(ln2s_atts, dim=0)
    }


class ml_datamodule(pl.LightningDataModule):  # noqa
    def __init__(self, tokenizer, ln1, ln2):
        super().__init__()
        self.tokenizer = tokenizer
        self.ln1 = ln1
        self.ln2 = ln2
        self.dataset = load_dataset('wmt19', f'{ln1}-{ln2}')
        # NOTE: assume that we are using the validation split
        self.length = len(self.dataset["validation"])

    @overrides
    def train_dataloader(self) -> DataLoader:
        return DataLoader(dataset=self.dataset["train"], num_workers=int(multiprocessing.cpu_count()/max(torch.cuda.device_count(), 1)),
            collate_fn=partial(collate_mm, tokenizer=self.tokenizer, ln1=self.ln1, ln2=self.ln2), batch_size=1)

    @overrides
    def val_dataloader(self) -> DataLoader:
        return DataLoader(dataset=self.dataset["validation"], num_workers=int(multiprocessing.cpu_count()/max(torch.cuda.device_count(), 1)), 
            collate_fn=partial(collate_mm, tokenizer=self.tokenizer, ln1=self.ln1, ln2=self.ln2), batch_size=1)

    @overrides
    def test_dataloader(self) -> DataLoader:
        return DataLoader(dataset=self.dataset["validation"], num_workers=int(multiprocessing.cpu_count()/max(torch.cuda.device_count(), 1)), 
            collate_fn=partial(collate_mm, tokenizer=self.tokenizer, ln1=self.ln1, ln2=self.ln2), batch_size=1)

def get_datamodule(mtype, ln1, ln2):
    if mtype == "mclip":
        tokenizer = transformers.AutoTokenizer.from_pretrained(AVAILABLE_MODELS['M-BERT-Base-69']['tokenizer_name'])
    elif mtype == "mbert":
        tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
    assert tokenizer
    return ml_datamodule(tokenizer, ln1, ln2)
