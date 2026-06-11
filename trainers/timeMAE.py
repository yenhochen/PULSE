import torch
import torch.nn as nn
import numpy as np

from trainers.base import BaseTrainer
from baselines.timeMAE.timeMAE import TimeMAE, Align, Reconstruct


class TimeMAETrainer(BaseTrainer):
    def __init__(self, config, train_data, val_data):
        super().__init__(config, train_data, val_data)

        self.model = TimeMAE(config).to(self.config.device)
        self.align = Align()
        self.reconstruct = Reconstruct()
        self.all_modules = {"model": self.model}
        self.model.to(self.config.device)

    def run_one_epoch(self, loader, train: bool):
        self.model.train(train)

        with torch.set_grad_enabled(train):
            epoch_loss = 0
            for batch in loader:
                self.optimizer.zero_grad()

                batch = batch.to(self.config.device)
                loss = self.run_one_batch(batch)
                if train:
                    loss.backward()
                    torch.nn.utils.clip_grad_value_(self.model.parameters(), 5)
                    self.optimizer.step()
                    self.scheduler.step()

                epoch_loss += loss.item()
            epoch_loss /= len(loader)

        return epoch_loss, dict()

    def run_one_batch(self, batch):
        """Args: batch (batch, time, channels). Returns scalar pretrain loss."""
        batch = batch.to(self.config.device)
        [rep_mask, rep_mask_prediction], [token_prediction_prob, tokens] = (
            self.model.pretrain_forward(batch)
        )

        align_loss = self.align.compute(rep_mask, rep_mask_prediction)
        reconstruct_loss, _, _ = self.reconstruct.compute(token_prediction_prob, tokens)

        loss = (
            self.config.model_args.alpha * align_loss
            + self.config.model_args.beta * reconstruct_loss
        )
        return loss

    def evaluate(self, dataloader, labels=None):
        with torch.no_grad():
            self.model.eval()
            results = {"embed": [], "labels": []}

            for batch in dataloader:
                if isinstance(batch, list):
                    batch, labels = batch

                embed = self.model(batch)
                results["embed"].append(embed.cpu())
                results["labels"].append(labels.cpu())

            results["embed"] = np.concatenate(results["embed"])
            results["labels"] = np.concatenate(results["labels"])
            return results

    def encode_downstream(self, batch):
        """Args: batch (batch, time, channels). Returns context embedding and None."""
        _, [context, _] = self.model.pretrain_forward(batch)
        return context, None

    def get_encoder(self):
        return TimeMAEFinetuneWrapper(self.model)


class TimeMAEFinetuneWrapper(nn.Module):
    """Wrap TimeMAE pretrain_forward for downstream encoding."""

    def __init__(self, model: TimeMAE):
        super().__init__()
        self.model = model

    def forward(self, x):
        """Args: x (batch, time, channels). Returns (context, None)."""
        _, [context, _] = self.model.pretrain_forward(x)
        return context, None
