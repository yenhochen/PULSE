import random
import torch
import torch.nn as nn
import numpy as np

from tqdm import tqdm
from pathlib import Path
from abc import abstractmethod
from pulse.encoder import TSEncoder
from utils.logging import get_logger
from utils.io import save_checkpoint

from utils.dataset import TimeSeriesDataset
from utils.constants import REQUIRES_LABELS
from torch.utils.data import DataLoader


logger = get_logger()


class BaseTrainer(nn.Module):
    def __init__(
        self,
        config,
        train_data,
        val_data,
    ):
        super(BaseTrainer, self).__init__()

        self.config = config
        self.train_data = train_data
        self.val_data = val_data
        self.model_type = config.model_type

        self.device = config.device
        self.batch_size = config.training_args.batch_size
        self.epochs = config.training_args.epochs
        self.subseq_size = config.data_args.subseq_size

        self.init_dl_program()

        self.encoder = TSEncoder(config)

        self.metrics_dict = {"train_loss": [], "val_loss": []}
        self.criterion = nn.MSELoss()

        train_dl_args = {"data": self.train_data, "train": True}
        val_dl_args = {"data": self.val_data, "train": False}

        if any(self.model_type == i for i in REQUIRES_LABELS):
            train_dl_args["labels"] = self.train_labels
            val_dl_args["labels"] = self.val_labels

        self.train_loader, self.val_loader = self.setup_dataloader(
            **train_dl_args
        ), self.setup_dataloader(**val_dl_args)
        self.optimizer = None
        self.scheduler = None

    def init_dl_program(
        self,
    ):
        max_threads = self.config.max_threads
        device_name = self.config.device

        if max_threads is not None:
            torch.set_num_threads(max_threads)  # intraop
            if torch.get_num_interop_threads() != max_threads:
                torch.set_num_interop_threads(max_threads)  # interop
            try:
                import mkl
            except:
                pass
            else:
                mkl.set_num_threads(max_threads)

        seed = self.config.seed
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

        if isinstance(device_name, (str, int)):
            device_name = [device_name]

        devices = []
        for t in reversed(device_name):
            t_device = torch.device(t)
            devices.append(t_device)
            if t_device.type == "cuda":
                assert torch.cuda.is_available()
                torch.cuda.set_device(t_device)
                if seed is not None:
                    seed += 1
                    torch.cuda.manual_seed(seed)
        devices.reverse()

    def setup_optimizer(self):
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.config.training_args.lr, weight_decay=1e-5
        )
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.config.training_args.lr,
            total_steps=len(self.train_loader) * self.config.training_args.epochs,
        )

    def setup_dataloader(self, data: np.array, train: bool, labels=None):
        """Build a windowed DataLoader from full time-series arrays."""
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

    @abstractmethod
    def run_one_epoch(self, dataloader: torch.utils.data.DataLoader): ...

    def evaluate(self, dataloader, labels=None):
        """Encode batches and return pooled embeddings with labels."""
        with torch.no_grad():
            self.model.eval()
            results = {"embed": [], "labels": []}

            for batch in dataloader:
                if isinstance(batch, list):
                    batch, labels = batch

                out, _ = self.encoder(batch)

                results["embed"].append(out.cpu())
                results["labels"].append(labels.cpu())

            results["embed"] = np.concatenate(results["embed"])
            results["labels"] = np.concatenate(results["labels"])
            return results

    def on_before_optimizer_step(self, optimizer):
        pass

    def fit(
        self,
    ):
        logger.info(f"Begin Training {self.model_type} SSL on seed {self.config.seed}")
        if self.optimizer is None:
            self.setup_optimizer()

        best_val_loss = np.inf

        pbar = tqdm(range(self.config.training_args.epochs))
        for epoch in pbar:
            self.current_epoch = epoch

            train_loss, train_postfix = self.run_one_epoch(self.train_loader, True)
            if epoch % self.config.training_args.eval_every_n == 0:
                val_loss, _ = self.run_one_epoch(self.val_loader, False)

            self.metrics_dict["train_loss"].append(train_loss)
            self.metrics_dict["val_loss"].append(val_loss)

            # save best checkpoint over trailing window of val losses
            n_size = 5
            if (
                len(self.metrics_dict["val_loss"][-n_size:]) >= n_size
                and np.mean(self.metrics_dict["val_loss"][-n_size:]) <= best_val_loss
            ):
                best_val_loss = val_loss
                save_checkpoint(
                    self,
                    Path(self.config.save_dir) / f"checkpoint_best",
                    additional_info={"epoch": epoch, "metrics": self.metrics_dict},
                )
                logger.info(
                    f"Saving best checkpoint at epoch {epoch} with val loss {val_loss:.5f}"
                )

            if (
                epoch % self.config.training_args.save_every_n == 0 and epoch != 0
            ):
                save_checkpoint(
                    self,
                    Path(self.config.save_dir) / f"checkpoint_{epoch}",
                    additional_info={"epoch": epoch, "metrics": self.metrics_dict},
                )
                logger.info(f"Saving checkpoint at epoch {epoch}")

            if epoch % self.config.training_args.log_every_n == 0:
                logger.info(
                    f"Epoch #{epoch}: train loss={train_loss:.5f}, val loss={val_loss:.5f}"
                )

            save_checkpoint(
                self,
                Path(self.config.save_dir) / f"checkpoint_last",
                additional_info={"epoch": epoch, "metrics": self.metrics_dict},
            )

            postfix = dict(
                loss=f"{train_loss:.5f}",
                val_loss=f"{val_loss:.5f}",
            )

            postfix.update(train_postfix)
            pbar.set_postfix(postfix)

    def load(self, ckpt_path=None, ckpt="best"):
        if ckpt_path is not None:
            ckpt_dir = ckpt_path
        else:
            ckpt_dir = self.run_dir

        loaded = torch.load(
            f"{ckpt_dir}/checkpoint_{ckpt}.pt",
            weights_only=False,
            map_location=self.config.device,
        )
        self.load_state_dict(loaded["state_dict"])

    def set_rundir(self, rundir):
        self.run_dir = rundir

    def get_encoder(
        self,
    ):
        return self.encoder

    def encode_downstream(self, batch):
        """Args: batch (batch, time, channels). Returns pooled and unpooled embeddings."""
        context_pool, context_all = self.encoder(batch)
        return context_pool, context_all
