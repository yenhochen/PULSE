import os
import json
import yaml
import torch
import pickle
import random
import numpy as np
from datetime import datetime
from pathlib import Path
from omegaconf import OmegaConf

from utils.logging import get_logger

logger = get_logger()


def find_all_directory_children_w_filter(top_path, filter):
    children = []
    for root, dirs, files in os.walk(top_path):
        for name in dirs + files:
            if filter in name:
                children.append(os.path.join(root, name))
    return children


def write_list_to_txtfile(input_list, save_path):
    """Save a list to a text file, one item per line."""
    with open(save_path, "w") as f:
        for item in input_list:
            f.write(f"{item}\n")


def write_json(data, path):
    with open(path, "w") as f:
        json.dump(data, f, indent=4)
    return path


def save_json(data, path):
    with open(path, "w") as f:
        json.dump(data, f, indent=4)
    return path


def get_date():
    return datetime.now().strftime("%Y%m%d")


def load_pickle(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data


def save_pickle(x, save_path):
    with open(save_path, "wb") as f:
        pickle.dump(x, f)


def load_yaml(yaml_path):
    with open(yaml_path, "r") as file:
        config = yaml.safe_load(file)
    return config


def save_yaml(x, yaml_path):
    with open(yaml_path, "w") as yaml_file:
        yaml.dump(x, yaml_file, default_flow_style=False, sort_keys=False)


def save_config(config, save_dir=None, fname="config.yaml"):
    if save_dir is None:
        save_dir = Path(config["save_dir"])

    config = OmegaConf.to_container(config, resolve=True)
    save_yaml(config, save_dir / fname)


def save_checkpoint(trainer, save_dir, additional_info={}):
    os.makedirs(save_dir, exist_ok=True)

    save_dir = Path(save_dir)
    torch.save(
        {k: v.state_dict() for k, v in trainer.all_modules.items()},
        save_dir / "model_state.pt",
    )
    torch.save(trainer.optimizer.state_dict(), save_dir / "optimizer.pt")
    torch.save(trainer.scheduler.state_dict(), save_dir / "scheduler.pt")
    save_config(trainer.config, save_dir)
    if len(additional_info):
        save_json(additional_info, save_dir / "info.json")


def get_full_config(config_path):
    """Merge configs from the given path and its base_config chain (root first)."""
    paths = [config_path]
    c = load_yaml(paths[-1])
    while "base_config" in c:
        base_config_path = c["base_config"]
        paths.append(base_config_path)
        c = load_yaml(paths[-1])

    configs = [OmegaConf.load(p) for p in paths[::-1]]
    config = OmegaConf.merge(*configs)
    return config


def resolve_save_dir(config, seed, *, transfer=False):
    """Default experiment output path when save_dir is null in config."""
    if config.get("save_dir") is not None:
        return config["save_dir"]

    data_name = [
        i for i in config["data_args"]["path"].split("/") if i != "data"
    ][0]
    subdir_name = (
        config["model_name"] if config.get("model_name") is not None else config["model_type"]
    )
    if transfer:
        return os.path.join(
            "experiments", "transfer", data_name, config["model_type"], f"seed_{seed}"
        )
    return os.path.join("experiments", data_name, subdir_name, f"seed_{seed}")


def load_trainer(config, data, checkpoint_name=None):
    import trainers.all_trainers as all_trainers
    from utils.dataset import get_trainer_kwargs

    load_dir = Path(config.save_dir)
    if checkpoint_name is not None:
        if (load_dir / checkpoint_name).exists():
            logger.info(
                f"Loading trainer from checkpoint: {load_dir / checkpoint_name}"
            )
            config_path = load_dir / checkpoint_name / f"config.yaml"
            config = get_full_config(config_path)
        else:
            logger.info(f"No checkpoint found at {load_dir / checkpoint_name}.")
            logger.info("Initializing new trainer from provided config.")
    else:
        logger.info("Initializing new trainer from provided config.")

    trainer_kwargs = get_trainer_kwargs(config, data)
    trainer = all_trainers.all_trainers[config.model_type](config, **trainer_kwargs)

    is_trained = False
    if checkpoint_name is not None and (load_dir / checkpoint_name).exists():
        is_trained = True
        logger.info("Loading model and optimizer state into trainer.")
        state_dict = torch.load(
            load_dir / checkpoint_name / f"model_state.pt",
            weights_only=False,
            map_location=trainer.config.device,
        )

        for k, v in state_dict.items():
            trainer.all_modules[k].load_state_dict(v)

        if config["model_type"] == "transfer":
            return trainer, is_trained
        trainer.setup_optimizer()

        optimizer_dict = torch.load(
            load_dir / checkpoint_name / f"optimizer.pt",
            weights_only=False,
            map_location=trainer.config.device,
        )
        scheduler_dict = torch.load(
            load_dir / checkpoint_name / f"scheduler.pt",
            weights_only=False,
            map_location=trainer.config.device,
        )
        trainer.optimizer.load_state_dict(optimizer_dict)
        trainer.scheduler.load_state_dict(scheduler_dict)

    return trainer, is_trained


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
