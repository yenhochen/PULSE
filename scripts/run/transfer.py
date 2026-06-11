"""Transfer-learning entry point: fine-tune a pretrained backbone on a target dataset."""

import argparse
import os
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from trainers.transfer import TransferTrainer
from utils.common import set_seed
from utils.constants import SPLIT_LIST
from utils.dataset import get_data_from_config, get_eval_dataloader
from utils.evaluate import eval_classification
from utils.io import get_full_config, save_pickle, save_yaml
from utils.logging import get_logger, log_class_scores


def get_parser():
    parser = argparse.ArgumentParser(
        description="Fine-tune a pretrained SSL checkpoint on a labeled target dataset."
    )
    parser.add_argument(
        "-c",
        "--config_path",
        required=True,
        help="Transfer experiment yaml (e.g. configs/transfer/epilepsy.yaml)",
    )
    parser.add_argument(
        "-p",
        "--ckpt_path",
        default=None,
        help="Pretrained checkpoint dir (overrides load_from_checkpoint in config)",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=None,
        help="Random seed (overrides config)",
    )
    return parser


def resolve_save_dir(config):
    if config.get("data_name"):
        data_name = config["data_name"]
    else:
        data_name = config["data_args"]["path"].split("/")[1]

    ckpt_path = Path(config["load_from_checkpoint"])
    seed = ckpt_path.parent.stem
    model_name = ckpt_path.parent.parent.stem
    return os.path.join("experiments", data_name, model_name, seed)


def run(config):
    data, _ = get_data_from_config(config, "subseq", train=True)

    assert config.load_from_checkpoint is not None, (
        "load_from_checkpoint must be set (use -p or config yaml)"
    )

    config.save_dir = resolve_save_dir(config)
    save_dir = Path(config.save_dir)
    os.makedirs(save_dir, exist_ok=True)

    logger = get_logger(log_file=save_dir / "log.log")
    logger.info(f"Log file created:\t{save_dir / 'log.log'}")
    logger.info(f"Loading data from:\t{config.data_args.path}")
    logger.info(f"train data shape:\t{data['train_data'].shape}")
    logger.info(f"val data shape:\t{data['val_data'].shape}")
    logger.info(f"loading from checkpoint:\t{config.load_from_checkpoint}")

    save_yaml(OmegaConf.to_container(config, resolve=True), save_dir / "config.yaml")

    train_data = np.concatenate([data["train_data"], data["val_data"]], axis=0)
    train_labels = np.concatenate([data["train_labels"], data["val_labels"]], axis=0)

    trainer = TransferTrainer(
        config,
        train_data,
        train_labels,
        data["val_data"],
        data["val_labels"],
    )
    trainer.fit()

    eval_loaders, _ = get_eval_dataloader(config)
    eval_results = {}
    for split in SPLIT_LIST:
        results = trainer.evaluate(eval_loaders[f"{split}_loader"])
        results["embed"] = results["embed"].squeeze()
        eval_results[f"{split}_results"] = results

    eval_class_results = eval_classification(
        eval_results["train_results"]["embed"],
        eval_results["train_results"]["labels"],
        eval_results["test_results"]["embed"],
        eval_results["test_results"]["labels"],
    )
    log_class_scores(eval_class_results)

    scores = {"classification": eval_class_results}
    score_path = save_dir / "scores.pkl"
    save_pickle(scores, score_path)
    logger.info(f"Saving scores to {score_path}")


if __name__ == "__main__":
    args = get_parser().parse_args()
    config = get_full_config(args.config_path)

    if args.seed is not None:
        config.seed = args.seed
    if args.ckpt_path is not None:
        config.load_from_checkpoint = args.ckpt_path

    set_seed(config.seed)
    run(config)
