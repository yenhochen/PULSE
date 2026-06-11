"""Re-run semi-supervised evaluation on a pretrained checkpoint."""

import argparse
from pathlib import Path

from utils.common import set_seed
from utils.constants import SPLIT_LIST
from utils.dataset import get_data_from_config, get_eval_dataloader
from utils.evaluate import eval_semisupervised
from utils.io import get_full_config, load_trainer, save_pickle
from utils.logging import get_logger, log_semisupervised


def get_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate frozen embeddings with 1% and 5% label fractions "
            "on an existing checkpoint_best directory."
        )
    )
    parser.add_argument(
        "-c",
        "--ckpt_path",
        required=True,
        help="Path to checkpoint dir (e.g. experiments/har/pulse/seed_0/checkpoint_best)",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=None,
        help="Override config seed",
    )
    parser.add_argument(
        "-n",
        "--n_samples",
        type=int,
        default=None,
        help="Override eval_args.eval_n_seeds (number of label subsampling draws)",
    )
    return parser


def run(config, ckpt_path):
    ckpt_path = Path(ckpt_path)
    save_dir = Path(config.save_dir)
    logger = get_logger(log_file=save_dir / "log.log")

    data, _ = get_data_from_config(config, config.data_args.mode)
    trainer, _ = load_trainer(config, data, "checkpoint_best")
    eval_loaders, _ = get_eval_dataloader(config)

    eval_results = {}
    for split in SPLIT_LIST:
        results = trainer.evaluate(eval_loaders[f"{split}_loader"])
        results["embed"] = results["embed"].squeeze()
        eval_results[f"{split}_results"] = results

    semisupervised_results = eval_semisupervised(
        config,
        eval_results["train_results"]["embed"],
        eval_results["train_results"]["labels"],
        eval_results["test_results"]["embed"],
        eval_results["test_results"]["labels"],
    )
    log_semisupervised(semisupervised_results)

    output_path = ckpt_path.parent / "semi_supervised_results.pkl"
    save_pickle(semisupervised_results, output_path)
    logger.info(f"Saving results to {output_path}")


if __name__ == "__main__":
    args = get_parser().parse_args()
    ckpt_path = Path(args.ckpt_path)

    config = get_full_config(ckpt_path / "config.yaml")
    if config.save_dir != str(ckpt_path.parent):
        config.save_dir = str(ckpt_path.parent)

    if args.seed is not None:
        config.seed = args.seed
    if args.n_samples is not None:
        config.eval_args.eval_n_seeds = args.n_samples

    set_seed(config.seed)
    run(config, ckpt_path)
