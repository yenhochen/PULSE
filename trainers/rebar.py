import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from trainers.base import BaseTrainer
from torch.utils.data import TensorDataset, DataLoader
from baselines.rebar.REBAR_CrossAttn import RebarCrossAttnTrainer


class RebarTrainer(BaseTrainer):
    def __init__(
        self,
        config,
        train_data=None,
        val_data=None,
    ):
        super().__init__(config, train_data, val_data)
        self.tau = config.model_args.tau
        self.alpha = config.model_args.alpha
        self.candidateset_size = config.model_args.candidateset_size

        ##### ----> Loading and training REBAR cross-attn to guide our picks of positives and negatives for contrastive learning <---- #####

        if not config.transfer:
            self.rebar_crossattn_trainer = RebarCrossAttnTrainer(
                config, train_data, val_data
            )
            self.rebar_crossattn_trainer.fit_rebarcrossattn()
            self.rebar_crossattn_trainer.load("best")

        self.encoder.to(self.config.device)
        self.model = self.encoder
        self.all_modules = {
            "encoder": self.encoder,
        }
        self.model_components = nn.ModuleDict(self.all_modules)
        self.model_components.to(config.device)

    def setup_dataloader(
        self, data: np.array, train: bool
    ) -> torch.utils.data.DataLoader:
        dataset = TensorDataset(torch.from_numpy(data).to(torch.float))
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=train,
            num_workers=torch.get_num_threads(),
        )
        return loader

    def run_one_epoch(self, dataloader: torch.utils.data.DataLoader, train: bool):
        self.model.train(mode=train)
        self.optimizer.zero_grad()
        with torch.set_grad_enabled(train):
            total_loss = 0
            for batch in dataloader:
                x = batch[0].to(self.device)
                bs, tslen, channels = x.shape
                t = np.random.randint(0, tslen - self.subseq_size)
                x_t_anchor = torch.clone(x[:, t : t + self.subseq_size, :])

                distances = []
                x_tc_candset = []
                for _ in range(self.candidateset_size):
                    tc = np.random.choice(
                        a=np.arange(
                            self.subseq_size // 2, tslen - 3 * self.subseq_size // 2
                        )
                    )
                    x_tc_cand = x[:, tc : tc + self.subseq_size, :]
                    x_tc_candset.append(x_tc_cand)
                    ##### ----> this is the key here!!! Using REBAR Cross Attention to calculate distance between distance and anchor <---- #####
                    distance = self.rebar_crossattn_trainer.calc_distance(
                        anchor=x_t_anchor, candidate=x_tc_cand
                    )
                    distances.append(distance)
                x_tc_candset = torch.cat(x_tc_candset).to(self.device)
                distances = torch.stack(distances)  # shape [candset_sizes, batch_size]
                labels = torch.argmin(
                    distances, dim=0
                )  # for a thing in the batch, we want best of cands, so should be length bs

                _, out1 = self.model(x_t_anchor)
                _, out2 = self.model(x_tc_candset)
                loss = contrastive_loss_imp(
                    z1=out1, z2=out2, labels=labels, alpha=self.alpha, tau=self.tau
                )
                loss /= bs
                if train:
                    loss.backward()
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                total_loss += loss.item()

            return total_loss, {}

    def encode_downstream(self, batch):
        """Args: batch (batch, time, channels). Returns pooled and unpooled embeddings."""
        context_pool, context_all = self.model(batch)
        return context_pool, context_all

    def get_encoder(self):
        return self.model


def contrastive_loss_imp(z1, z2, labels, tau=1, alpha=0.5):
    # labels is length BS, bc it tells us which of the cand samples is the best
    # z1 shape [BS, length, channels]
    # z2 shape [BS*candset_size, length, channels]
    z1 = F.max_pool1d(
        z1.transpose(1, 2).contiguous(), kernel_size=z1.size(1)
    ).transpose(1, 2)
    z2 = F.max_pool1d(
        z2.transpose(1, 2).contiguous(), kernel_size=z2.size(1)
    ).transpose(1, 2)

    loss = instance_contrastive_loss_imp(z1, z2, labels, tau=tau)
    loss *= alpha
    loss += (1 - alpha) * temporal_contrastive_loss_imp(z1, z2, labels, tau=tau)
    return loss.to(device=z1.device)


def instance_contrastive_loss_imp(z1, z2, labels, tau=1):
    # for a given time, other stuff in the batch is negative
    # z1 shape [BS, length, channels]
    # z2 shape [BS*candset_size, length, channels]

    # need to get this T x 2B x 2B
    (
        bs,
        ts_len,
        channels,
    ) = z1.shape
    candset_size = z2.shape[0] // bs

    loss = torch.zeros(bs, device=z1.device)
    for batch_idx in range(bs):
        # [1 x channel] x [channel x candset_size]
        # I want a 1 x candset_size
        temp_z1 = z1[batch_idx, :].contiguous().view(1, -1)
        # for batch_idx 3, we know the 4th mc is the best, so to get there. we go to the 4th mc by doing 4*bs and then going to the + batch idx
        positive = z2[labels[batch_idx] * bs + batch_idx, :].contiguous().view(1, -1)
        negatives = torch.cat(
            (
                z1[:batch_idx, :].contiguous().view(-1, positive.shape[-1]),
                z1[batch_idx + 1 :, :].contiguous().view(-1, positive.shape[-1]),
            )
        )
        temp_z2 = torch.cat((positive, negatives))

        sim = F.cosine_similarity(temp_z1, temp_z2, dim=1).unsqueeze(0)
        logits = -F.log_softmax(sim / tau, dim=-1)
        loss[batch_idx] = logits[0, 0]

    return loss.mean()


def temporal_contrastive_loss_imp(z1, z2, labels, tau=1):
    # z1 shape [BS, length, channels]
    # z2 shape [BS*candset_size, length, channels]
    (
        bs,
        ts_len,
        channels,
    ) = z1.shape
    candset_size = z2.shape[0] // bs

    loss = torch.zeros(bs, device=z1.device)
    for batch_idx in range(bs):
        # with time as a dimension so you could do it by, this means first time must be the same as the other time step
        # [1 x length*channel] x [length*channel x candset_size]
        # better way could be cosine similarity with a set
        # [1 x channel] x [channel x candset_size]
        # I want a 1 x candset_size
        temp_z1 = z1[batch_idx, :].contiguous().view(1, -1)
        temp_z2 = (
            z2[batch_idx::bs, :].contiguous().view(candset_size, -1)
        )  # positive is batch_idx + bs*(labels[batch_idx])

        sim = F.cosine_similarity(temp_z1, temp_z2, dim=1).unsqueeze(0)
        logits = -F.log_softmax(sim / tau, dim=-1)
        loss[batch_idx] = logits[0, labels[batch_idx]]

    return loss.mean()

