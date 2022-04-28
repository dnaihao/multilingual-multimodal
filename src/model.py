import pytorch_lightning as pl 
import torch

from overrides import overrides
from typing import Optional, Any, Mapping


class FineTuner(pl.LightningModule):

    def __init__(self, model, mtype, ln1, ln2, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.model = model
        self.stats = {ln1: [], ln2: []}
        self.mtype = mtype
        self.ln1 = ln1
        self.ln2 = ln2
    
    @overrides(check_signature=False)
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> Any:
        kwargs = {"attention_mask": attention_mask}
        return self.model(input_ids, **kwargs)

    def _step(self, batch, split):
        if split == "test":
            if self.mtype == "mbert":
                ln1_outputs = self(input_ids=batch[self.ln1], attention_mask=batch[f"{self.ln1}_attention_mask"])["last_hidden_state"][0]
                ln2_outputs = self(input_ids=batch[self.ln2], attention_mask=batch[f"{self.ln2}_attention_mask"])["last_hidden_state"][0]
                self.stats[self.ln1].append(ln1_outputs.unsqueeze(0).sum(dim=1)/ln1_outputs.shape[1])
                self.stats[self.ln2].append(ln2_outputs.unsqueeze(0).sum(dim=1)/ln2_outputs.shape[1])           
            elif self.mtype == "mclip":
                ln1_outputs = self(input_ids=batch[self.ln1], attention_mask=batch[f"{self.ln1}_attention_mask"])[0]
                ln2_outputs = self(input_ids=batch[self.ln2], attention_mask=batch[f"{self.ln2}_attention_mask"])[0]
                self.stats[self.ln1].append(ln1_outputs.unsqueeze(0).sum(dim=1)/ln1_outputs.shape[1])
                self.stats[self.ln2].append(ln2_outputs.unsqueeze(0).sum(dim=1)/ln2_outputs.shape[1])
        return ln1_outputs, ln2_outputs

    @overrides(check_signature=False)
    def training_step(self, batch: Mapping[str, Any], batch_idx: int = 0) -> torch.Tensor:
        return self._step(batch, split="train")

    @overrides(check_signature=False)
    def validation_step(self, batch: Mapping[str, Any], batch_idx) -> None:
        return self._step(batch, split="val")

    @overrides(check_signature=False)
    def test_step(self, batch: Mapping[str, Any], batch_idx) -> None:
        return self._step(batch, split="test")