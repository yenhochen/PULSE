import torch
import numpy as np

from einops import rearrange
from trainers.base import BaseTrainer
from torch.utils.data import DataLoader
from baselines.dsvae.dsvae import DSVAE
from utils.dataset import TimeSeriesDataset


class DSVAETrainer(BaseTrainer):
    def __init__(self, config, train_data, val_data):
        super().__init__(config, train_data, val_data)

        config.model_args.device = config.device
        config.model_args.x_dim = config.data_args.input_dims

        self.model = DSVAE(config.model_args, self.encoder)

        self.all_modules = {"model": self.model}
        self.model.to(self.config.device)

    def run_one_epoch(self, loader, train: bool):
        self.model.train(train)

        with torch.set_grad_enabled(train):
            epoch_loss, epoch_recon = 0, 0
            for batch in loader:
                self.optimizer.zero_grad()

                batch = batch.to(self.config.device)
                loss, y, context = self.run_one_batch(
                    batch,
                )
                if train:
                    loss.backward()

                    torch.nn.utils.clip_grad_value_(self.model.parameters(), 5)

                    self.optimizer.step()
                    self.scheduler.step()

                epoch_loss += loss.item()
            epoch_loss /= len(loader)

        return epoch_loss, dict()

    def run_one_batch(self, batch):
        batch = batch.to(self.config.device)
        x = rearrange(batch, "b t c -> t b c")
        y, context = self.model.forward(x)

        loss_kl_z = loss_KLD(
            self.model.z_mean,
            self.model.z_logvar,
            self.model.z_mean_p,
            self.model.z_logvar_p,
        )
        loss_kl_v = loss_KLD(
            self.model.v_mean,
            self.model.v_logvar,
            self.model.v_mean_p,
            self.model.v_logvar_p,
        )
        loss_kl = loss_kl_z + loss_kl_v

        loss_recon = torch.nn.functional.mse_loss(y, x, reduction="mean")

        loss = loss_recon + self.config.model_args.kl_weight * loss_kl

        y = rearrange(y, "t b c -> b t c")

        return loss, y, context

    def encode_downstream(self, batch):
        context_pool, context_all = self.encoder(batch)
        return context_pool, context_all

    def get_encoder(
        self,
    ):
        return self.encoder


def loss_KLD(z_mean, z_logvar, z_mean_p=0, z_logvar_p=0):
    ret = -0.5 * torch.sum(
        z_logvar
        - z_logvar_p
        - torch.div(
            z_logvar.exp() + (z_mean - z_mean_p).pow(2), z_logvar_p.exp() + 1e-10
        )
    )
    return ret

