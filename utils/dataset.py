"""Dataset loading and preprocessing for time-series experiments.

Supports two data modes (data_args.mode):
  subseq  - load pre-cut windows from {split}_data_subseq.npy
  fullts  - load full series from {split}_data.npy, wrap into windows at train time

For analysis experiments, data_args.analysis_args triggers AnalysisDataset which
samples n_classes attractor trajectories per seed.

Training subsampling is controlled by p_train / p_val and keyed to config.seed.
"""

import os
import json
import torch
import numpy as np
import numpy.random as npr
from torch.utils.data import DataLoader, TensorDataset, Dataset
from pathlib import Path

# from utils.logging import printlog
from utils.logging import get_logger

logger = get_logger()

from utils.constants import SPLIT_LIST, REQUIRES_LABELS

# SPLIT_LIST = ["train", "val", "test"]
# REQUIRES_LABELS = ["supervised", "pulse_oracle"]


def load_data(config, data_type):
    data = {
        "train_data": None,
        "train_labels": None,
        "val_data": None,
        "val_labels": None,
        "test_data": None,
        "test_labels": None,
    }

    if "analysis_args" in config["data_args"]:
        config["data_args"]["mode"] = data_type
        dataset = AnalysisDataset(config)
        return dataset.data, dataset

    data_path = Path(config["data_args"]["path"])

    if data_type == "fullts":
        annotate = ""
    elif data_type == "subseq":
        annotate = "_subseq"
    else:
        print("data_type must be subseq or fullts")
        import sys

        sys.exit()

    # load labels
    for split in SPLIT_LIST:
        if os.path.exists(data_path / f"{split}_labels{annotate}.npy"):
            data[f"{split}_labels"] = np.load(
                data_path / f"{split}_labels{annotate}.npy"
            )
        data[f"{split}_data"] = np.load(data_path / f"{split}_data{annotate}.npy")
    return data, {}


def get_data_from_config(config, data_type, train=False):
    data, info = load_data(config, data_type)

    # select channels
    if config.data_args.include_channels != "all":
        for split in SPLIT_LIST:
            data[f"{split}_data"] = data[f"{split}_data"][
                :, :, config.data_args.include_channels
            ]

    # downsample over time
    for split in SPLIT_LIST:
        data[f"{split}_data"] = data[f"{split}_data"][
            :, :: config.data_args.downsample_factor
        ]

    if train:
        data = subsample_over_batch(
            config, data, data_type
        )  # sample_subsequences, for efficient training and debugging

    # if config.data_args.input_dims != data["train_data"].shape[-1]:
    # logger.info("Setting input dimensions in config based on the data.")
    # config.data_args.input_dims = data["train_data"].shape[-1] # set input dims in config based on the data

    return data, info


def subsample_over_batch(config, data, data_type):

    subseq_size = config.data_args.subseq_size
    if (
        data_type == "fullts" and config.model_type != "rebar"
    ):  # do not subsample for rebar, since it takes in full time series
        data["train_data"] = subsample_fullts_data(
            data["train_data"],
            config.data_args.p_train,
            subseq_size,
            skip=config.data_args.train_stride,
            seed=config.seed,
        )
        data["val_data"] = subsample_fullts_data(
            data["val_data"],
            config.data_args.p_val,
            subseq_size,
            skip=config.data_args.val_stride,
            seed=config.seed,
        )

    if data_type == "subseq":
        data["train_data"], train_ixs = subsample_subseq_data(
            data["train_data"], config.data_args.p_train, seed=config.seed
        )
        data["val_data"], val_ixs = subsample_subseq_data(
            data["val_data"], config.data_args.p_val, seed=config.seed
        )

        data["train_labels"] = data["train_labels"][train_ixs]
        data["val_labels"] = data["val_labels"][val_ixs]

    return data


def subsample_subseq_data(data, p, seed=1234):
    # print(data.shape, p)
    # randomly keep subsequences from the data.
    n_keep = int(p * len(data))
    np.random.seed(seed)
    ixs = np.arange(len(data))
    ixs = np.random.permutation(ixs)[:n_keep]
    return data[ixs], ixs


def subsample_fullts_data(data, percent, subseq_size, skip=64, seed=1234):
    # randomly sample subsequences from the full time series data.
    sampled_dataset = []
    for i in range(len(data)):  # iterate over trials.
        npr.seed(seed + i)
        ixs = np.arange(len(data[i]) - subseq_size - 2)
        ixs = npr.permutation(ixs)[::skip]
        t = len(ixs)
        ixs = ixs[: int(t * percent)]
        time_indices = np.expand_dims(torch.arange(0, subseq_size), 0) + np.expand_dims(
            ixs, 1
        )
        sampled_dataset.append(data[i, time_indices, :])

    sampled_dataset = np.concatenate(sampled_dataset)
    return sampled_dataset


def get_eval_dataloader(config):
    data, data_info = get_data_from_config(config, "subseq")

    out = {}
    for split in SPLIT_LIST:
        x = data[f"{split}_data"]
        x = x[
            :, : config.data_args.subseq_size
        ]  # truncate subseq data to speciofied length

        print(x.shape, data[f"{split}_labels"].shape)

        # if config.model_type == "pulse_oracle":
        #     dataset = TensorDataset(torch.from_numpy(x).to(config.device).float(),
        #                             torch.from_numpy(data[f"{split}_labels"]).to(config.device).long(),
        #                             torch.from_numpy(np.arange(len(x))).to(config.device).long(),
        #                             )

        # else:
        dataset = TensorDataset(
            torch.from_numpy(x).to(config.device).float(),
            torch.from_numpy(data[f"{split}_labels"]).to(config.device).long(),
        )

        out[f"{split}_labels"] = data[f"{split}_labels"]
        out[f"{split}_loader"] = DataLoader(
            dataset, batch_size=config.training_args.eval_batch_size, shuffle=False
        )

    return out, data_info


def get_label_names(config):
    """
    load the label names from the json file located in processed data directory
    """
    # data_path = get_data_path(config)
    data_path = Path(config.data_args.path)

    if "analysis_args" in config["data_args"]:
        # config["data_args"]["mode"] = data_type
        dataset = AnalysisDataset(config, verbose=False)
        return dataset.label_names
        # return dataset.data, dataset

    with open(data_path / "label_name.json", "r") as file:
        label_names = json.load(file)

    return {int(k): v for k, v in label_names.items()}  # convert keys to int


def get_trainer_kwargs(config, data):
    trainer_kwargs = {
        "train_data": data["train_data"],
        "val_data": data["val_data"],
    }
    if config.model_type in REQUIRES_LABELS:
        trainer_kwargs.update(
            {"train_labels": data["train_labels"], "val_labels": data["val_labels"]}
        )
    return trainer_kwargs


def data_to_subseq(data, labels, subseq_len=300):
    """
    Convert data to subsequences of length subseq_len.
    """
    b, t, c = data.shape
    s = (
        t // subseq_len
    )  # number of segments. drop last segment if t is not a multiple of subseq_len
    s = int(t / subseq_len)  # number of segments
    labels_subseq = np.repeat(labels, s)
    return (
        data[:, : s * subseq_len].reshape(b * s, subseq_len, c),
        labels_subseq,
    )  # reshape to (b*s, subseq_len, c), (b*s,)


class AnalysisDataset:
    def __init__(self, config, verbose=True):
        self.config = config
        self.verbose = verbose

        self.data = {
            "train_data": [],
            "train_labels": [],
            "val_data": [],
            "val_labels": [],
            "test_data": [],
            "test_labels": [],
        }

        self.paths = []
        self.label_names = {}
        self.build_data()

    def build_data(
        self,
    ):
        """
        retrieve analysis_config and then randomly select n_classes
        """
        analysis_config = self.config["data_args"]["analysis_args"]
        analysis_config["subseq_size"] = self.config["data_args"]["subseq_size"]

        # subdirs = [i for i in os.listdir(analysis_config["data_dir"]) if os.path.isdir(i)]

        data_dir = Path(self.config["data_args"]["path"])
        subdirs = [
            data_dir / i for i in os.listdir(data_dir) if os.path.isdir(data_dir / i)
        ]

        # randomly select n_classes subdirectories based on seed
        np.random.seed(self.config["seed"])  # set seed for reproducibility
        subdirs = np.random.permutation(subdirs)[: analysis_config["n_classes"]]
        if self.verbose:
            logger.info("Loading analysis subdirectories:")
            [logger.info(f"\t{s}") for s in subdirs]
        [self.paths.append(s) for s in subdirs]

        for k, subdir in enumerate(subdirs):
            analysis_config["data_args"] = {}
            analysis_config["data_args"]["path"] = subdir
            analysis_data, _ = load_data(analysis_config, "fullts")
            for split in SPLIT_LIST:
                x = analysis_data[f"{split}_data"]
                labels = np.ones(len(x), dtype=int) * k

                if self.config["data_args"]["mode"] == "subseq":
                    x, labels = data_to_subseq(
                        x, labels, analysis_config["subseq_size"]
                    )

                self.data[f"{split}_data"].append(x)
                self.data[f"{split}_labels"].append(labels)  # dummy labels for now

            self.label_names[k] = subdir.name
        analysis_config.pop("data_args", None)

        logger.info("")

        for k in self.data.keys():
            self.data[k] = np.concatenate(self.data[k])


class TimeSeriesDataset(Dataset):
    """
    A PyTorch Dataset for time series data that returns windows of a given size with a specified stride.
    If there are left over samples, they will be discarded.

    Args:
        data (torch.Tensor): Input tensor of shape (b, t, c).
        window_size (int): Size of the sliding window.
        stride (int): Step size for the sliding window.
    """

    def __init__(
        self,
        data,
        window_size,
        stride,
        #  config,
        labels=None,
        #  augment=True
    ):

        # self.config = config

        self.data = data
        self.window_size = window_size
        self.stride = stride

        # self.labels = labels

        # Check dimensions
        if self.data.ndim != 3:
            raise ValueError("Input data must have 3 dimensions (b, t, c)")

        self.b, self.t, self.c = self.data.shape

        print(self.b, self.t, self.c, self.window_size, self.stride)

        # Calculate the number of windows per sequence
        self.num_windows = (self.t - self.window_size) // self.stride + 1
        if self.num_windows <= 0:
            raise ValueError(
                "Window size and stride are incompatible with sequence length"
            )

    def __len__(self):
        # Total number of windows across all sequences
        return self.b * self.num_windows

    def __getitem__(self, idx):
        # Map global index to batch and window indices
        batch_idx = idx // self.num_windows
        window_idx = idx % self.num_windows

        # Calculate start and end indices for the window
        start = window_idx * self.stride
        end = start + self.window_size

        return self.data[batch_idx, start:end, :]

