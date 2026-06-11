"""Generate per-parameter config.yaml files for synthetic SDE datasets."""

import argparse
import os
from pathlib import Path

import numpy as np

from utils.io import load_yaml, save_yaml

ANALYSIS_ROOT = Path(__file__).resolve().parent
GENERATE_DATA_COMMANDS = ANALYSIS_ROOT / "generate_data_commands.sh"


def get_parser():
    parser = argparse.ArgumentParser(
        description="Expand a base analysis config into per-parameter config.yaml files."
    )
    parser.add_argument(
        "-c",
        "--config_path",
        help="Path to base analysis yaml (e.g. configs/lorenz-base.yaml)",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-n",
        "--noise",
        help="process_noise_std (overrides config sde_args.diffusion_args.sigma)",
        type=str,
        required=True,
    )
    return parser


def generate_configs(config):
    """Write configs under data/analysis/{dynamics_fn}/noise-{sigma}/{tag}/config.yaml."""
    if config["save_dir"] is None:
        config["save_dir"] = "data/analysis"

    save_dir = (config["save_dir"] + ".")[:-1]

    sweep_values = config["sweep_values"].copy()
    sweep_key = config["sweep_key"].split("/").copy()

    config["save_dir"] = None
    config.pop("sweep_values", None)
    config.pop("sweep_key", None)

    speed = np.linspace(
        config["speed_start"], config["speed_end"], num=len(sweep_values)
    )

    config_paths = []
    for seed, (sv, sp) in enumerate(zip(sweep_values, speed)):
        d = config
        for key in sweep_key[:-1]:
            d = d[key]
        d[sweep_key[-1]] = sv

        tag = "_".join(
            [
                f"{k}-{v}".replace("/", "-")
                for k, v in config["sde_args"]["dynamics_args"].items()
            ]
        )
        noise = config["sde_args"]["diffusion_args"]["sigma"]
        noise_str = f"noise-{noise}"

        config["sde_args"]["speed"] = sp.item()
        config["seed"] = seed

        save_dir_ = Path(save_dir) / config["sde_args"]["dynamics_fn"] / noise_str / tag
        os.makedirs(save_dir_, exist_ok=True)

        save_config_path = save_dir_ / "config.yaml"
        save_yaml(config, save_config_path)

        config_paths.append(save_config_path)
        print(f"Config saved to: {save_config_path}")

    with open(GENERATE_DATA_COMMANDS, "a") as f:
        for config_path in config_paths:
            f.write(
                f"python data/process/analysis/generate_data.py --config_path {config_path}\n"
            )

    print(f"Appended {len(config_paths)} commands to: {GENERATE_DATA_COMMANDS}")


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    config_path = Path(args.config_path)
    if not config_path.is_absolute():
        config_path = Path.cwd() / config_path

    config = load_yaml(config_path)
    config["sde_args"]["diffusion_args"]["sigma"] = float(args.noise)

    generate_configs(config)
