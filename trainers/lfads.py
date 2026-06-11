import torch
import numpy as np

from torch.utils.data import DataLoader
from typing import Any, Dict, Union
from utils.dataset import TimeSeriesDataset
from trainers.base import BaseTrainer
from baselines.lfads.lfads import LFADS
from baselines.lfads.metrics import r2_score
from baselines.lfads.l2 import compute_l2_penalty_2


class LFADSTrainer(BaseTrainer):
    def __init__(self, config, train_data, val_data):
        super().__init__(config, train_data, val_data)

        config["encod_data_dim"] = config["data_args"]["input_dims"]
        config["encod_seq_len"] = config["data_args"]["subseq_size"]
        config["recon_seq_len"] = config["data_args"]["subseq_size"]

        self.model = LFADS(config, self.encoder)
        self.all_modules = {"encoder": self.encoder, "model": self.model}
        self.model.to(self.config.device)

    def setup_dataloader(self, data: np.array, train: bool, labels=None):
        stride = (
            self.config.data_args.train_stride
            if train
            else self.config.data_args.val_stride
        )
        if stride is None:
            stride = 1

        dataset = TimeSeriesDataset(
            torch.from_numpy(data).to(torch.float),
            self.config.data_args.subseq_size,
            stride,
            labels=labels,
        )

        loader = DataLoader(
            dataset,
            batch_size=self.config.training_args.batch_size,
            shuffle=train,
            num_workers=torch.get_num_threads(),
        )
        return loader

    def run_one_epoch(self, loader, train: bool):
        self.model.train(train)

        with torch.set_grad_enabled(train):
            epoch_loss = 0
            for batch in loader:
                self.optimizer.zero_grad()

                loss, output, recon = self.run_one_batch(batch, train)

                if train:
                    loss.backward()
                    torch.nn.utils.clip_grad_value_(self.model.parameters(), 1)
                    self.on_before_optimizer_step(self.optimizer)
                    self.optimizer.step()
                    self.scheduler.step(loss)

                epoch_loss += loss.item()

            epoch_loss /= len(loader)

        return epoch_loss, {"recon": recon.item()}

    def run_one_batch(self, batch, train: bool) -> torch.Tensor:
        """Args: batch (batch, time, channels). Returns loss, LFADS output dict, recon scalar."""
        batch = batch.to(self.config.device)

        aug_stack = self.model.train_aug_stack if train else self.model.infer_aug_stack
        batch = aug_stack.process_batch(batch)
        output = self.model.forward(
            batch, sample_posteriors=self.config.variational, output_means=False
        )

        recon_all = self.model.recon.compute_loss(batch, output["output_params"])
        recon_all = aug_stack.process_losses(recon_all, batch)

        if not self.config.recon_reduce_mean:
            recon_all = torch.sum(recon_all, dim=(1, 2))

        recon = recon_all.mean()
        l2 = compute_l2_penalty_2(self.model, self.config)

        ic_mean = output["ic_mean"]
        ic_std = output["ic_std"]
        co_means = output["co_means"]
        co_stds = output["co_stds"]

        ic_kl = self.model.ic_prior(ic_mean, ic_std) * self.config.kl_ic_scale
        co_kl = self.model.co_prior(co_means, co_stds) * self.config.kl_co_scale

        l2_ramp = self._compute_ramp(
            self.config.l2_start_epoch, self.config.l2_increase_epoch
        )
        kl_ramp = self._compute_ramp(
            self.config.kl_start_epoch, self.config.kl_increase_epoch
        )

        loss = self.config.loss_scale * (
            recon + l2_ramp * l2 + kl_ramp * (ic_kl + co_kl)
        )

        output_means = self.model.recon.compute_means(output["output_params"])
        r2 = torch.mean(torch.stack([r2_score(output_means, batch)]))
        return loss, output, recon

    def _compute_ramp(self, start: int, increase: int):
        """Ramp coefficient from 0 to 1 over `increase` epochs starting at `start`."""
        ramp = (self.current_epoch + 1 - start) / (increase + 1)
        return torch.clamp(torch.tensor(ramp), 0, 1)

    def on_before_optimizer_step(self, optimizer: torch.optim.Optimizer):
        """Gradually ramp weight decay alongside L2 regularization."""
        l2_ramp = self._compute_ramp(
            self.config.l2_start_epoch, self.config.l2_increase_epoch
        )
        optimizer.param_groups[0]["weight_decay"] = l2_ramp * self.config.weight_decay

    def setup_optimizer(self) -> Union[torch.optim.Optimizer, Dict[str, Any]]:
        hps = self.config
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=hps.training_args.lr,
            weight_decay=hps.weight_decay,
        )

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=self.optimizer,
            mode="min",
            factor=0.95,
            patience=6,
            threshold=0.0,
            min_lr=1e-5,
        )
