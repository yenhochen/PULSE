"""Fine-tune a frozen SSL backbone on a labeled target dataset (linear probe heads)."""

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import TensorDataset, DataLoader

from trainers.base import BaseTrainer
from pulse.mlp import MLP
from utils.io import get_full_config
from utils.dataset import get_data_from_config, get_trainer_kwargs
from utils.logging import get_logger

logger = get_logger()


class TransferTrainer(BaseTrainer):
    """Load a pretrained encoder checkpoint and train input/output heads on target labels."""

    def __init__(self, config, train_data, train_labels, val_data, val_labels):
        self.train_labels = train_labels
        self.val_labels = val_labels
        super().__init__(config, train_data, val_data)

        self.ckpt_trainer, self.ckpt_config = load_transfer_backbone_ckpt(config)
        self.load_heads(config, self.ckpt_config)
        self.encoder = self.ckpt_trainer.get_encoder()

        self.all_modules = {
            "encoder": self.encoder,
            "in_head": self.in_head,
            "out_head": self.out_head,
        }
        self.model = nn.ModuleDict(self.all_modules)
        self.model.to(self.config.device)

        self.criterion = nn.CrossEntropyLoss()
        self.current_epoch = 0
        self.freeze_module(self.encoder)

    def run_one_batch(self, batch):
        """Args: batch tuple (windows, labels). Returns loss, logits, embeddings."""
        batch, labels = batch
        batch = batch.to(self.config.device)
        labels = labels.to(self.config.device).long()

        x = self.in_head(batch)
        embed, _ = self.encoder(x)
        pred_logit = self.out_head(embed)
        loss = self.criterion(pred_logit, labels)
        return loss, pred_logit, embed

    def run_one_epoch(self, loader, train):
        if self.current_epoch == self.config.training_args.freeze_backbone_epochs:
            logger.info(f"Unfreezing backbone at epoch {self.current_epoch}")
            self.unfreeze_module(self.encoder)

        self.model.train(train)
        with torch.set_grad_enabled(train):
            epoch_loss = 0
            for batch in loader:
                if train:
                    self.optimizer.zero_grad()

                loss, _, _ = self.run_one_batch(batch)

                if train:
                    loss.backward()
                    self.optimizer.step()
                    self.scheduler.step()

                epoch_loss += loss.item()

            epoch_loss /= len(loader)

        self.current_epoch += 1
        return epoch_loss, {}

    def setup_dataloader(self, data, labels, train: bool):
        dataset = TensorDataset(
            torch.from_numpy(data).to(torch.float),
            torch.from_numpy(labels).to(torch.float),
        )
        return DataLoader(
            dataset,
            batch_size=self.config.training_args.batch_size,
            shuffle=train,
            num_workers=torch.get_num_threads(),
        )

    def load_heads(self, config, ckpt_config):
        if config.data_args.input_dims == ckpt_config.data_args.input_dims:
            self.in_head = nn.Identity()
        else:
            self.in_head = nn.Linear(
                config.data_args.input_dims, ckpt_config.data_args.input_dims
            )

        self.out_head = MLP(
            ckpt_config.encoder_args.emb_dim,
            64,
            config.data_args.num_classes,
            activation=nn.Tanh(),
        )

    def freeze_module(self, module):
        for param in module.parameters():
            param.requires_grad = False

    def unfreeze_module(self, module):
        for param in module.parameters():
            param.requires_grad = True

    def evaluate(self, loader):
        with torch.no_grad():
            self.model.eval()
            results = {
                "embed": [],
                "labels": [],
                "pred_proba": [],
                "pred_labels": [],
            }
            for batch in loader:
                _, labels = batch
                _, out, context = self.run_one_batch(batch)

                probs = torch.softmax(out.cpu(), dim=-1)
                preds = torch.argmax(probs, dim=-1)

                results["pred_proba"].append(probs)
                results["pred_labels"].append(preds)
                results["embed"].append(context.cpu())
                results["labels"].append(labels.cpu())

            for key in results:
                results[key] = np.concatenate(results[key])
            return results

    def setup_optimizer(self):
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.config.training_args.lr, weight_decay=1e-6
        )
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.optimizer, gamma=0.9999
        )


def load_transfer_backbone_ckpt(config):
    """Load pretrained backbone weights from checkpoint_best directory."""
    import trainers.all_trainers as all_trainers

    ckpt_dir = Path(config.load_from_checkpoint)
    ckpt_config = get_full_config(ckpt_dir / "config.yaml")
    ckpt_data, _ = get_data_from_config(ckpt_config, ckpt_config.data_args.mode)
    ckpt_trainer_kwargs = get_trainer_kwargs(ckpt_config, ckpt_data)

    ckpt_trainer = all_trainers.all_trainers[ckpt_config.model_type](
        ckpt_config, **ckpt_trainer_kwargs
    )
    state_dict = torch.load(
        ckpt_dir / "model_state.pt",
        weights_only=False,
        map_location=ckpt_trainer.config.device,
    )

    for name, _ in state_dict.items():
        logger.info(f"Loading weights for {name}")
        ckpt_trainer.all_modules[name].load_state_dict(state_dict[name])

    return ckpt_trainer, ckpt_config
