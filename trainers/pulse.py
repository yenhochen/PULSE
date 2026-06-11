"""PULSE trainer: predictive SSL with init-condition reconstruction.

Training loop (run_one_batch):
  1. Encode input with shared TSEncoder
  2. Extract optional time-varying latent (TimeVaryingModule)
  3. Predict GRU initial state from input window (InitConditionEncoder)
  4. Reconstruct future timesteps (ReconstructionNet)
  5. Minimize MSE between prediction and ground-truth future

See configs/base/pulse_base.yaml for hyperparameter documentation.
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader

from pulse.augment import DynamicAugmentations
from pulse.timeVarying import TimeVaryingModule
from pulse.reconstruct import ReconstructionNet
from pulse.initialCondition import InitConditionEncoder, SharedInitConditionEncoder
from utils.common import shift_and_mask, get_pred_true
from utils.dataset import TimeSeriesDataset
from trainers.base import BaseTrainer


class PULSETrainer(BaseTrainer):
    """Self-supervised PULSE pretraining with multi-view augmentation."""

    def __init__(self, config, train_data, val_data):
        super(PULSETrainer, self).__init__(config, train_data, val_data)

        self.context_norm = config.encoder_args.norm_last_layer
        self.standardize_batch = config.training_args.standardize_batch

        if config.model_args.shared_f_init:
            self.init_encoder = SharedInitConditionEncoder(config)
        else:
            self.init_encoder = InitConditionEncoder(config)

        self.recon_net = ReconstructionNet(config)
        self.aug = DynamicAugmentations(config)

        if config.model_args.time_vary_args.include:
            self.tv_module = TimeVaryingModule(config)

        self.all_modules = {
            "encoder": self.encoder,
            "init_encoder": self.init_encoder,
            "recon_net": self.recon_net,
            "aug": self.aug,
            "tv_module": (
                self.tv_module
                if config.model_args.time_vary_args.include
                else nn.Identity()
            ),
        }

        self.model = nn.ModuleDict(self.all_modules)
        self.model.to(self.config.device)

        self.dropout = nn.Dropout(0.1)

    def setup_dataloader(self, data: np.array, train: bool, labels=None):
        stride = (
            self.config.data_args.train_stride
            if train
            else self.config.data_args.val_stride
        )

        dataset = TimeSeriesDataset(
            torch.from_numpy(data).to(torch.float),
            self.config.data_args.subseq_size,
            stride,
            labels=labels,
        )

        g = torch.Generator()
        g.manual_seed(self.config.seed)

        loader = DataLoader(
            dataset,
            batch_size=self.config.training_args.batch_size,
            shuffle=train,
            num_workers=torch.get_num_threads(),
            generator=g,
            worker_init_fn=lambda _: np.random.seed(self.config.seed),
        )
        return loader

    def get_timevarying(self, context):
        """Args: context (batch, time, emb_dim). Returns tv, dtv or (None, None)."""
        if self.config.model_args.time_vary_args.include:
            tv, dtv = self.tv_module(context)
        else:
            tv, dtv = (None, None)

        return tv, dtv

    def run_one_batch(self, batch, sample_init=False):
        """Args: batch (batch, time, channels). Returns pred, true, and diagnostics tuple."""
        batch = batch.to(self.config.device)
        batch_ = batch.clone()  # input: (batch, time, channels)

        context, context_unpooled = self.encoder(batch_)
        # context: (batch, emb_dim); context_unpooled: (batch, time, emb_dim)
        tv, dtv = self.get_timevarying(context_unpooled)

        if self.config.model_args.shared_f_init: # for ablation
            h0, start_ix, n_steps = self.init_encoder(
                context_unpooled,
                sample_init=sample_init,
                sample_right_boundary=self.config.model_args.augmentation_args.sample_right_boundary,
            )  # h0: (num_layers, batch, hidden_dim)
        else:
            h0, start_ix, n_steps = self.init_encoder(
                batch_,
                sample_init=sample_init,
                sample_right_boundary=self.config.model_args.augmentation_args.sample_right_boundary,
            )  # h0: (num_layers, batch, hidden_dim)

        recon_inputs = self.aug.get_recon_inputs(context, n_steps.max())
        # recon_inputs: (batch, n_steps, emb_dim)

        dtv, m = (
            shift_and_mask(dtv, start_ix)
            if self.config.model_args.time_vary_args.include
            else (dtv, None)
        )
        recon_inputs = (
            torch.dstack([recon_inputs, dtv]).contiguous()
            if self.config.model_args.time_vary_args.include
            else recon_inputs
        )

        recon_inputs = self.dropout(recon_inputs)

        out, hs = self.recon_net(recon_inputs, h0)
        # out: (batch, time, channels)
        true, pred = get_pred_true(
            batch_, out, sample_init=sample_init, start_ix=start_ix
        )

        return pred, true, (out, h0, hs, context, start_ix, dtv)

    def run_one_epoch(self, loader, train: bool):
        self.model.train(train)

        with torch.set_grad_enabled(train):
            epoch_loss, epoch_recon = 0, 0
            for batch in loader:

                self.optimizer.zero_grad()

                x_anchor = batch.to(self.config.device)

                loss_sample = 0
                n_samples = self.config.model_args.augmentation_args.n_init_samples
                if n_samples > 0:
                    for i in range(n_samples):
                        pred, true, (out, x0, hs, context, start_ix, cdtv) = (
                            self.run_one_batch(x_anchor, sample_init=True)
                        )
                        loss_sample += self.criterion(pred, true)

                    loss_sample /= n_samples

                pred, true, (out, x0, hs, context, start_ix, cdtv) = self.run_one_batch(
                    x_anchor, sample_init=False
                )
                loss_nosample = self.criterion(pred, true)

                loss_recon = 0.5 * (loss_sample + loss_nosample)
                loss = loss_recon

                if train:
                    loss.backward()
                    torch.nn.utils.clip_grad_value_(self.model.parameters(), 5)
                    self.optimizer.step()
                    self.scheduler.step()

                epoch_loss += loss.item()
                epoch_recon += loss_recon.item()

            epoch_loss /= len(loader)
            epoch_recon /= len(loader)

        return epoch_loss, dict(
            h0_max=f"{torch.abs(x0[0]).max():.4f}",
            context_max=f"{torch.abs(context).max():.4f}",
            out_max=f"{torch.abs(pred).max():.4f}",
        )

    def evaluate(self, dataloader):
        with torch.no_grad():
            self.model.eval()
            results = {"pred": [], "true": [], "embed": [], "labels": []}
            for batch in dataloader:
                if isinstance(batch, list):
                    batch, labels = batch

                pred, true, (out, x0, hs, embed_pooled, start_ix, cdtv) = (
                    self.run_one_batch(batch, sample_init=False)
                )

                results["pred"].append(pred.cpu())
                results["true"].append(true.cpu())
                results["embed"].append(embed_pooled.cpu())
                results["labels"].append(labels.cpu())

            results["pred"] = np.concatenate(results["pred"])
            results["true"] = np.concatenate(results["true"])
            results["embed"] = np.concatenate(results["embed"])
            results["labels"] = np.concatenate(results["labels"])

            return results

    def encode_downstream(self, batch):
        """Args: batch (batch, time, channels). Returns pooled and unpooled embeddings."""
        context_pool, context_all = self.encoder(batch)
        return context_pool, context_all

    def encode_init(self, batch):
        """Args: batch (batch, time, channels). Returns init projection (batch, time, hidden)."""
        x0_all = self.init_encoder.init_proj(batch)
        return x0_all

    def get_encoder(self):
        return self.encoder
