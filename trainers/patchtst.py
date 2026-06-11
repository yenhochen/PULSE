"""PatchTST self-supervised pretraining via Hugging Face transformers."""

import numpy as np
import torch
import torch.nn as nn

from trainers.base import BaseTrainer
from transformers import PatchTSTConfig, PatchTSTForPretraining
from transformers.models.patchtst.modeling_patchtst import PatchTSTMasking


class PatchTSTTrainer(BaseTrainer):
    """Masked-patch pretraining with PatchTST; downstream eval uses CLS token embeddings."""

    def __init__(self, config, train_data=None, val_data=None):
        super().__init__(config, train_data, val_data)

        self.model = PatchTSTWrapper(config)
        self.all_modules = {"encoder": self.model}
        self.model.to(self.config.device)

    def run_one_epoch(self, dataloader, train: bool):
        # Val loss is still masked pretraining loss; masking only off for downstream evaluate().
        self.model.enable_masking()

        self.optimizer.zero_grad()
        total_loss = 0

        for batch in dataloader:
            batch = batch.to(self.config.device)
            b, _, _ = batch.shape

            out = self.model.train_forward(batch)
            loss = out.loss / b

            if train:
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

            total_loss += loss.item()

        return total_loss, {}

    def evaluate(self, dataloader, labels=None):
        with torch.no_grad():
            self.model.disable_masking()
            self.model.eval()
            results = {"embed": [], "labels": []}

            for batch in dataloader:
                if isinstance(batch, list):
                    batch, labels = batch

                out = self.model.infer_forward(batch.to(self.config.device))
                results["embed"].append(out.cpu())
                results["labels"].append(labels.cpu())

            results["embed"] = np.concatenate(results["embed"])
            results["labels"] = np.concatenate(results["labels"])
            return results


class PatchTSTWrapper(nn.Module):
    """Thin wrapper: builds PatchTSTConfig from yaml and exposes train/infer forwards."""

    def __init__(self, config):
        super().__init__()
        self.patchtst_config = PatchTSTConfig(
            num_input_channels=config.data_args.input_dims,
            context_length=config.data_args.subseq_size,
            **config.model_args,
        )

        self.model = PatchTSTForPretraining(self.patchtst_config)
        self.backbone = self.model.model
        self.flatten = nn.Flatten(start_dim=1, end_dim=-1)

    def enable_masking(self):
        self.model.train()
        self.model.model.do_mask_input = True
        self.model.model.masking = PatchTSTMasking(self.patchtst_config)

    def disable_masking(self):
        self.model.eval()
        self.model.model.do_mask_input = False
        self.model.model.masking = nn.Identity()

    def infer_forward(self, batch):
        """Args: batch (batch, time, channels). Returns (batch, emb_dim) CLS embeddings."""
        out = self.backbone(past_values=batch)
        cls_tokens = out.last_hidden_state[:, :, 0, :]
        return self.flatten(cls_tokens)

    def train_forward(self, batch):
        """Args: batch (batch, time, channels). Returns PatchTST pretraining output with .loss."""
        return self.model(past_values=batch)
