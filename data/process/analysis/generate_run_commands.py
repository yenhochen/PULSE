"""Generate pretrain commands for synthetic analysis experiments."""

import argparse
import os
from pathlib import Path

from utils.experiment_commands import RUN_PRETRAIN
from utils.io import (
    find_all_directory_children_w_filter,
    write_list_to_txtfile,
    get_full_config,
)
from utils.logging import get_logger

logger = get_logger()

DEFAULT_SAVE = Path("scripts/generated/commands_analysis.sh")


def get_parser():
    parser = argparse.ArgumentParser(
        description="Build scripts/run/pretrain.py commands for all analysis data × model configs."
    )
    parser.add_argument(
        "-c",
        "--config_path",
        help="Directory of experiment yaml configs (e.g. configs/analysis-w-100)",
        type=str,
        required=True,
    )
    parser.add_argument("-n", "--n_seeds", help="Number of random seeds", type=int, default=10)
    parser.add_argument(
        "-s",
        "--save_name",
        help="Output shell script path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "-r",
        "--retrain",
        help="Include commands even when scores.pkl already exists",
        type=bool,
        default=False,
    )
    return parser


def run(args):
    config_paths = find_all_directory_children_w_filter(args.config_path, ".yaml")
    data_paths = find_all_directory_children_w_filter("data/analysis", "noise")

    commands = []
    for c_path in config_paths:
        logger.info(f"building commands for: {c_path}")
        config = get_full_config(c_path)

        for path in data_paths:
            for seed in range(args.n_seeds):
                subdir_name = config["model_type"]
                if config["model_name"] is not None:
                    subdir_name = config["model_name"]

                save_dir = (
                    Path("experiments/analysis/")
                    / path.split("/")[2]
                    / f"{Path(path).stem}"
                    / subdir_name
                    / f"seed-{seed}"
                )
                cmd = (
                    f"{RUN_PRETRAIN} -c {c_path} -p {path} -s {seed} -sd {save_dir}"
                )
                if os.path.exists(save_dir / "scores.pkl") and not args.retrain:
                    logger.info(f"skipping {save_dir}. already exists.")
                    continue

                commands.append(cmd)

    save_name = args.save_name if args.save_name else str(DEFAULT_SAVE)
    logger.info(f"Writing command list to: {save_name}")
    write_list_to_txtfile(commands, save_name)


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    run(args)
