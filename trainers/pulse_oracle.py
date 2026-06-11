"""PULSE oracle trainer: reconstruction from paired same-class samples."""

import torch
import torch.nn as nn
import numpy as np

from torch.utils.data import DataLoader, TensorDataset

from pulse.augment import DynamicAugmentations
from pulse.timeVarying import TimeVaryingModule
from pulse.reconstruct import ReconstructionNet
from pulse.initialCondition import InitConditionEncoder, SharedInitConditionEncoder
from utils.common import shift_and_mask, get_pred_true
from trainers.base import BaseTrainer


class PULSEOracleTrainer(BaseTrainer):
    """Oracle PULSE pretraining using same-class paired windows as reconstruction targets."""

    def __init__(
        self,
        config,
        train_data,
        val_data,
        train_labels,
        val_labels,
    ):
        self.train_labels = train_labels
        self.val_labels = val_labels
        super(PULSEOracleTrainer, self).__init__(config, train_data, val_data)

        self.context_norm = config.encoder_args.norm_last_layer
        self.standardize_batch = config.training_args.standardize_batch

        if config.model_args.shared_f_init:
            self.init_encoder = SharedInitConditionEncoder(config)
        else:
            self.init_encoder = InitConditionEncoder(config)

        self.recon_net = ReconstructionNet(config)
        self.aug = DynamicAugmentations(config)
        self.dropout = (
            nn.Dropout1d(self.config.model_args.dropout_rate)
            if self.config.model_args.dropout_rate > 0
            else nn.Identity()
        )

        if config.model_args.time_vary_args.include:
            self.tv_module = TimeVaryingModule(config)

        self.all_modules = {
            "encoder": self.encoder,
            "init_encoder": self.init_encoder,
            "recon_net": self.recon_net,
            "aug": self.aug,
            "tv_module": (
                self.tv_module if config.model_args.time_vary_args.include else None
            ),
        }

        self.model = nn.ModuleDict(self.all_modules)
        self.model.to(self.config.device)

    def setup_dataloader(self, data, labels, train):
        ixs = np.arange(len(labels))
        dataset = TensorDataset(
            torch.from_numpy(data).to(torch.float),
            torch.from_numpy(labels).to(torch.long),
            torch.from_numpy(ixs).to(torch.long),
        )

        loader = DataLoader(
            dataset,
            batch_size=self.config.training_args.batch_size,
            shuffle=train,
            num_workers=torch.get_num_threads(),
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
        """Args: batch tuple of (windows, labels, indices). Returns pred, true, diagnostics."""
        batch, labels, ix = batch

        batch = batch.to(self.config.device)
        batch_ = batch.clone()  # input: (batch, time, channels)

        pairs_ixs = self.get_pairs_ixs(ix)
        assert (self.train_labels[ix] == self.train_labels[pairs_ixs]).any()
        batch_specific = torch.Tensor(self.train_data[pairs_ixs]).to(
            self.config.device
        )  # (batch, time, channels)

        if self.config.model_args.combine_inputs:
            batch_true = torch.cat([batch_, batch_specific], dim=-1)  # (batch, time, 2*channels)
            batch_ = self.dropout(batch_)
            batch_specific = self.dropout(batch_specific)
            batch_ = torch.cat([batch_, batch_specific], dim=-1)  # (batch, time, 2*channels)
            context, context_unpooled = self.encoder(batch_)
            tv, dtv = self.get_timevarying(context_unpooled)
        else:
            context, _ = self.encoder(batch_)  # (batch, emb_dim)
            _, context_unpooled = self.encoder(batch_specific)  # (batch, time, emb_dim)
            tv, dtv = self.get_timevarying(context_unpooled)

        if self.config.model_args.shared_f_init:
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

        out, hs = self.recon_net(recon_inputs, h0)

        if not self.config.model_args.combine_inputs:
            batch_true = batch_specific

        true, pred = get_pred_true(
            batch_true,
            out,
            start_ix=start_ix,
            sample_init=sample_init,
        )

        return pred, true, (out, h0, hs, context, start_ix, dtv)

    def run_one_epoch(self, loader, train: bool):
        self.model.train(train)

        with torch.set_grad_enabled(train):
            epoch_loss = 0
            for batch in loader:
                self.optimizer.zero_grad()
                pred, true, (out, x0, hs, context, start_ix, cdtv) = self.run_one_batch(
                    batch, sample_init=self.config.model_args.sample_init
                )
                loss = self.criterion(pred, true)

                if train:
                    loss.backward()
                    torch.nn.utils.clip_grad_value_(self.model.parameters(), 5)
                    self.optimizer.step()
                    self.scheduler.step()

                epoch_loss += loss.item()

            epoch_loss /= len(loader)

        return epoch_loss, dict(
            h0_max=f"{torch.abs(x0[0]).max():.4f}",
            context_max=f"{torch.abs(context).max():.4f}",
            out_max=f"{torch.abs(pred).max():.4f}",
        )

    def encode_downstream(self, batch):
        """Args: batch (batch, time, channels). Returns pooled and unpooled embeddings."""
        context_pool, context_all = self.encoder(batch)
        return context_pool, context_all

    def encode_init(self, batch):
        """Args: batch (batch, time, channels). Returns init projection (batch, time, hidden)."""
        x0_all = self.init_encoder.init_proj(batch)
        return x0_all

    def get_pairs_ixs(self, ix, seed=None):
        """Args: ix (batch,) sample indices. Returns paired indices from same class."""
        np.random.seed(seed)

        class_ixs = get_unique_labels_ix(self.train_labels)
        batch_class_ixs = get_unique_labels_ix(self.train_labels[ix])

        pairs_ixs = {}
        for u in np.unique(self.train_labels[ix]):
            candidate_set = np.random.choice(
                class_ixs[u], replace=False, size=len(batch_class_ixs[u]) * 2
            ).tolist()

            for i in ix[batch_class_ixs[u]]:
                c = candidate_set.pop()
                if i == c:
                    c = candidate_set.pop()
                pairs_ixs[i.item()] = c

        pairs_ixs = np.array([pairs_ixs[i.item()] for i in ix])
        return pairs_ixs

    def evaluate(self, dataloader, labels=None):
        """Encode batches; duplicate channels when combine_inputs is enabled."""
        with torch.no_grad():
            self.model.eval()
            results = {"embed": [], "labels": []}

            for batch in dataloader:
                if isinstance(batch, list):
                    batch, labels = batch

                if self.config.model_args.combine_inputs:
                    batch = torch.cat([batch, batch], dim=-1)

                out, _ = self.encoder(batch)

                results["embed"].append(out.cpu())
                results["labels"].append(labels.cpu())

            results["embed"] = np.concatenate(results["embed"])
            results["labels"] = np.concatenate(results["labels"])
            return results


def get_unique_labels_ix(arr):
    """Map each label value to the indices where it appears."""
    u = np.unique(arr)
    class_ixs = {i: [] for i in u}

    for idx, val in enumerate(arr):
        class_ixs[val].append(idx)

    class_ixs = {k: np.array(v) for k, v in class_ixs.items()}
    return class_ixs
