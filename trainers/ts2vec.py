from trainers.base import BaseTrainer
from torch.utils.data import TensorDataset, DataLoader

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class TS2VecTrainer(BaseTrainer):
    def __init__(self, config, train_data=None, val_data=None):
        self.max_train_length = config.data_args.subseq_size
        super().__init__(config, train_data, val_data)

        self.temporal_unit = config.model_args.temporal_unit

        self.model = self.encoder
        self.all_modules = {"encoder": self.encoder}
        self.model.to(self.config.device)

    def setup_dataloader(
        self, data: np.array, train: bool
    ) -> torch.utils.data.DataLoader:
        sections = data.shape[1] // self.max_train_length
        data = np.concatenate(split_with_nan(data, sections, axis=1), axis=0)

        np.random.seed(self.config.seed)
        ixs = np.arange(len(data))
        ixs = np.random.permutation(ixs)
        ixs = ixs[: int(self.config.data_args.p_train * len(ixs))]
        data = data[ixs]

        dataset = TensorDataset(torch.from_numpy(data).to(torch.float))
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=train,
            num_workers=torch.get_num_threads(),
        )

        return loader

    def run_one_epoch(self, dataloader: torch.utils.data.DataLoader, train: bool):
        self.encoder.train(mode=train)
        self.optimizer.zero_grad()

        with torch.set_grad_enabled(train):
            total_loss = 0

            for batch in dataloader:

                x = batch[0]
                if (
                    self.max_train_length is not None
                    and x.size(1) > self.max_train_length
                ):
                    window_offset = np.random.randint(
                        x.size(1) - self.max_train_length + 1
                    )
                    x = x[:, window_offset : window_offset + self.max_train_length]
                x = x.to(self.device)

                bs, ts_l, feat = x.shape
                crop_l = np.random.randint(
                    low=2 ** (self.temporal_unit + 1), high=ts_l + 1
                )
                crop_left = np.random.randint(ts_l - crop_l + 1)
                crop_right = crop_left + crop_l
                crop_eleft = np.random.randint(crop_left + 1)
                crop_eright = np.random.randint(low=crop_right, high=ts_l + 1)
                crop_offset = np.random.randint(
                    low=-crop_eleft, high=ts_l - crop_eright + 1, size=x.size(0)
                )

                out1 = self.model.ts_encoder(
                    take_per_row(x, crop_offset + crop_eleft, crop_right - crop_eleft),
                    mask="binomial",
                )
                out1 = out1[:, -crop_l:]

                out2 = self.model.ts_encoder(
                    take_per_row(x, crop_offset + crop_left, crop_eright - crop_left),
                    mask="binomial",
                )
                out2 = out2[:, :crop_l]

                loss = hierarchical_contrastive_loss(
                    out1,
                    out2,
                    temporal_unit=self.temporal_unit,
                )

                loss /= bs
                if train:
                    loss.backward()
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                total_loss += loss.item()

            return total_loss, {}

    def get_encoder(self):
        return self.model

    def encode_downstream(self, batch):
        """Args: batch (batch, time, channels). Returns pooled and unpooled TS2Vec embeddings."""
        context_all = self.model.ts_encoder(batch)
        context_pool = (
            torch.nn.functional.max_pool1d(
                context_all.transpose(1, 2), kernel_size=context_all.size(1)
            )
            .transpose(1, 2)
            .squeeze(1)
        )
        return context_pool, context_all


def hierarchical_contrastive_loss(z1, z2, alpha=0.5, temporal_unit=0):
    loss = torch.tensor(0.0, device=z1.device)
    d = 0
    while z1.size(1) > 1:
        if alpha != 0:
            loss += alpha * instance_contrastive_loss(z1, z2)
        if d >= temporal_unit:
            if 1 - alpha != 0:
                loss += (1 - alpha) * temporal_contrastive_loss(z1, z2)
        d += 1
        z1 = F.max_pool1d(z1.transpose(1, 2), kernel_size=2).transpose(1, 2)
        z2 = F.max_pool1d(z2.transpose(1, 2), kernel_size=2).transpose(1, 2)
    if z1.size(1) == 1:
        if alpha != 0:
            loss += alpha * instance_contrastive_loss(z1, z2)
        d += 1
    return loss / d


def instance_contrastive_loss(z1, z2):
    B, T = z1.size(0), z1.size(1)
    if B == 1:
        return z1.new_tensor(0.0)
    z = torch.cat([z1, z2], dim=0)  # 2B x T x C
    z = z.transpose(0, 1)  # T x 2B x C
    sim = torch.matmul(z, z.transpose(1, 2))  # T x 2B x 2B
    logits = torch.tril(sim, diagonal=-1)[:, :, :-1]  # T x 2B x (2B-1)
    logits += torch.triu(sim, diagonal=1)[:, :, 1:]
    logits = -F.log_softmax(logits, dim=-1)

    i = torch.arange(B, device=z1.device)
    loss = (logits[:, i, B + i - 1].mean() + logits[:, B + i, i].mean()) / 2
    return loss


def temporal_contrastive_loss(z1, z2):
    B, T = z1.size(0), z1.size(1)
    if T == 1:
        return z1.new_tensor(0.0)
    z = torch.cat([z1, z2], dim=1)  # B x 2T x C
    sim = torch.matmul(z, z.transpose(1, 2))  # B x 2T x 2T
    logits = torch.tril(sim, diagonal=-1)[:, :, :-1]  # B x 2T x (2T-1)
    logits += torch.triu(sim, diagonal=1)[:, :, 1:]
    logits = -F.log_softmax(logits, dim=-1)

    t = torch.arange(T, device=z1.device)
    loss = (logits[:, t, T + t - 1].mean() + logits[:, T + t, t].mean()) / 2
    return loss


def split_with_nan(x, sections, axis=0):
    assert x.dtype in [np.float16, np.float32, np.float64]
    arrs = np.array_split(x, sections, axis=axis)
    target_length = arrs[0].shape[axis]
    for i in range(len(arrs)):
        arrs[i] = pad_nan_to_target(arrs[i], target_length, axis=axis)
    return arrs


def pad_nan_to_target(array, target_length, axis=0, both_side=False):
    assert array.dtype in [np.float16, np.float32, np.float64]
    pad_size = target_length - array.shape[axis]
    if pad_size <= 0:
        return array
    npad = [(0, 0)] * array.ndim
    if both_side:
        npad[axis] = (pad_size // 2, pad_size - pad_size // 2)
    else:
        npad[axis] = (0, pad_size)
    return np.pad(array, pad_width=npad, mode="constant", constant_values=np.nan)


def take_per_row(A, indx, num_elem):
    all_indx = indx[:, None] + np.arange(num_elem)
    return A[torch.arange(all_indx.shape[0])[:, None], all_indx]
