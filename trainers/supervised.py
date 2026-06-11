import torch
import torch.nn as nn
import numpy as np

from trainers.base import BaseTrainer
from utils.logging import get_logger
from utils.dataset import get_label_names
from torch.utils.data import DataLoader, TensorDataset

logger = get_logger()


def load_matching_state_dict(model, pretrained_state_dict):
    model_state_dict = model.state_dict()
    matched_state_dict = {}

    for name, param in pretrained_state_dict.items():
        if name in model_state_dict and model_state_dict[name].shape == param.shape:
            matched_state_dict[name] = param

    # Load the matching parameters
    model_state_dict.update(matched_state_dict)
    model.load_state_dict(model_state_dict)

    print(f"Loaded {len(matched_state_dict)} / {len(model_state_dict)} parameters.")


class SupervisedTrainer(BaseTrainer):
    def __init__(self, config, train_data, train_labels, val_data, val_labels):
        self.train_labels = train_labels
        self.val_labels = val_labels

        super().__init__(config, train_data, val_data)

        self.model = self.encoder

        label_names = get_label_names(config)
        self.config.num_classes = len(label_names)

        self.model.to(self.config.device)

        if config.load_from_checkpoint is not None:
            print(f"Loading pretrained weights from {config.load_from_checkpoint}")
            exp_loaded = torch.load(
                config.load_from_checkpoint,
                map_location=self.config.device,
                weights_only=False,
            )
            load_matching_state_dict(self.model, exp_loaded["state_dict"])

        self.criterion = torch.nn.CrossEntropyLoss()
        self.classifier = nn.Linear(
            self.config.encoder_args.emb_dim, self.config.num_classes
        )
        self.classifier.to(self.config.device)

        self.all_modules = {
            "encoder": self.encoder,
            "classifier": self.classifier,
        }
        self.model_components = nn.ModuleDict(self.all_modules)
        self.model_components.to(config.device)

    def run_one_batch(self, batch):
        """Args: batch (windows, labels). Returns loss, logits (batch, num_classes), embed (batch, emb_dim)."""
        x, labels = batch
        x = x.to(self.config.device)
        labels = labels.to(self.config.device).long()

        embed, _ = self.model(x)
        pred_logit = self.classifier(embed)
        loss = self.criterion(pred_logit, labels)

        return loss, pred_logit, embed

    def run_one_epoch(self, loader, train):
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

        return epoch_loss, dict()

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

                loss, out, context = self.run_one_batch(
                    batch,
                )

                probs = torch.softmax(out.cpu(), dim=-1)
                preds = torch.argmax(probs, dim=-1)

                results["pred_proba"].append(probs)
                results["pred_labels"].append(preds)
                results["embed"].append(context.cpu())
                results["labels"].append(labels.cpu())

            results["pred_proba"] = np.concatenate(results["pred_proba"])
            results["pred_labels"] = np.concatenate(results["pred_labels"])
            results["embed"] = np.concatenate(results["embed"])
            results["labels"] = np.concatenate(results["labels"])

        return results

    def setup_dataloader(self, data, labels, train):

        dataset = TensorDataset(
            torch.from_numpy(data).to(torch.float),
            torch.from_numpy(labels).to(torch.long),
        )

        loader = DataLoader(
            dataset,
            batch_size=self.config.training_args.batch_size,
            shuffle=train,
            num_workers=torch.get_num_threads(),
        )
        return loader

    def encode_downstream(self, batch):
        """Args: batch (batch, time, channels). Returns pooled embedding (batch, emb_dim)."""
        context_pool, _ = self.model(batch)
        return context_pool

    def get_encoder(self):
        return self.model

