"""Shared helpers for generating experiment shell commands."""

import os
from pathlib import Path

from utils.io import (
    find_all_directory_children_w_filter,
    get_full_config,
    resolve_save_dir,
    write_list_to_txtfile,
)

DEFAULT_SEEDS = (0, 1, 2, 3, 4)
GENERATED_DIR = Path("scripts/generated")

RUN_PRETRAIN = "python scripts/run/pretrain.py"
RUN_TRANSFER = "python scripts/run/transfer.py"
RUN_SEMISUPERVISED = "python scripts/run/semisupervised.py"


def format_run_pretrain(config_path, seed, save_dir, epochs=None):
    cmd = f"{RUN_PRETRAIN} -c {config_path} -s {seed} -sd {save_dir}"
    if epochs is not None:
        cmd += f" -e {epochs}"
    return cmd


def format_run_transfer(transfer_config, ckpt_path, seed):
    return f"{RUN_TRANSFER} -c {transfer_config} -p {ckpt_path} -s {seed}"


def format_run_semisupervised(ckpt_path):
    return f"{RUN_SEMISUPERVISED} -c {ckpt_path}"


def write_command_script(commands, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_list_to_txtfile(commands, str(output_path))
    return str(output_path)


def resolve_linear_probe_save_dir(config, config_path, seed, save_root):
    if save_root is None:
        return resolve_save_dir(config, seed)

    model_name = Path(config_path).stem
    dataset = Path(config_path).parent.name
    return os.path.join(save_root, dataset, model_name, f"seed_{seed}")


def build_pretrain_commands(
    config_dir,
    seeds=DEFAULT_SEEDS,
    *,
    transfer=False,
    epochs=None,
    force=False,
    skip_marker="scores.pkl",
):
    """Build pretrain commands for all yaml configs under config_dir."""
    config_paths = sorted(find_all_directory_children_w_filter(config_dir, ".yaml"))
    commands = []

    for config_path in config_paths:
        for seed in seeds:
            config = get_full_config(config_path)
            save_dir = resolve_save_dir(config, seed, transfer=transfer)
            marker = Path(save_dir) / skip_marker

            if not force and marker.exists():
                continue

            commands.append(format_run_pretrain(config_path, seed, save_dir, epochs))

    return commands


def build_linear_probe_commands(
    top_config_dir="configs/linear_probe",
    seeds=DEFAULT_SEEDS,
    epochs=None,
    save_root=None,
    force=False,
    dataset=None,
    exclude_models=None,
):
    """Build pretrain commands for linear-probe configs."""
    config_paths = sorted(
        find_all_directory_children_w_filter(top_config_dir, ".yaml")
    )
    if dataset is not None:
        config_paths = [p for p in config_paths if Path(p).parent.name == dataset]
    if exclude_models:
        excluded = set(exclude_models)
        config_paths = [p for p in config_paths if Path(p).stem not in excluded]

    commands = []
    for config_path in config_paths:
        for seed in seeds:
            config = get_full_config(config_path)
            save_dir = resolve_linear_probe_save_dir(
                config, config_path, seed, save_root
            )

            if not force and (Path(save_dir) / "scores.pkl").exists():
                continue

            commands.append(
                format_run_pretrain(config_path, seed, save_dir, epochs)
            )

    return commands


def build_transfer_commands(top_backbone_dir, transfer_config):
    """Build fine-tuning commands from pretrained checkpoint directories."""
    paths = find_all_directory_children_w_filter(top_backbone_dir, "seed")
    paths = [p for p in paths if "hp_" not in p]

    commands = []
    for path in paths:
        ckpt_path = Path(path) / "checkpoint_best"
        if not ckpt_path.exists():
            continue
        seed = Path(path).stem.split("_")[-1]
        commands.append(format_run_transfer(transfer_config, ckpt_path, seed))

    return commands


def build_semisupervised_eval_commands(top_backbone_dir):
    """Build semi-supervised re-eval commands from checkpoint directories."""
    paths = find_all_directory_children_w_filter(top_backbone_dir, "seed")
    commands = []
    for path in paths:
        ckpt_path = Path(path) / "checkpoint_best"
        if ckpt_path.exists():
            commands.append(format_run_semisupervised(ckpt_path))
    return commands
