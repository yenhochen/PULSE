"""Simulate SDE trajectories and save train/val/test splits for analysis experiments."""

from fractions import Fraction
import os

import numpy as np
from utils.io import load_yaml
from pathlib import Path
from data.process.analysis.sdes import DYNAMICS_FNS, DIFFUSION_FNS, BaseSDE
import torch
import torchsde
import matplotlib.pyplot as plt

import sys

sys.setrecursionlimit(5000)

import argparse


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config_path", help="analysis yaml config", type=str, default=None
    )
    return parser

    # parser.add_argument('--yaml_path', )
    # args = parser.parse_args()


def generate_fullts(args):
    config = load_yaml(args.config_path)

    dynamics_args = {
        k: float(Fraction(v)) for k, v in config["sde_args"]["dynamics_args"].items()
    }  # convert to float
    dynamics_fn = DYNAMICS_FNS[config["sde_args"]["dynamics_fn"]](**dynamics_args)

    diffusion_clean_fn = DIFFUSION_FNS[config["sde_args"]["diffusion_fn"]](
        state_size=config["sde_args"]["diffusion_args"]["state_size"], sigma=0
    )

    config["generate_args"]["z_score"] = False
    clean_sde = BaseSDE(
        dynamics_fn, diffusion_clean_fn, speed=config["sde_args"]["speed"]
    )
    clean_data, y0_clean = generate_sde(
        clean_sde, seed=config["seed"], **config["generate_args"]
    )

    s = config["sde_args"]["diffusion_args"]["sigma"]
    sigma = ((clean_data - clean_data.mean((0, 1))) ** 2).norm(
        dim=(2)
    ).mean().sqrt() * s  # rms of the clean data
    sigma = sigma / config["sde_args"]["gamma"]  # scale factor dependent on system

    diffusion_fn = DIFFUSION_FNS[config["sde_args"]["diffusion_fn"]](
        state_size=config["sde_args"]["diffusion_args"]["state_size"], sigma=sigma
    )
    config["generate_args"]["z_score"] = True

    sde = BaseSDE(dynamics_fn, diffusion_fn, speed=config["sde_args"]["speed"])
    data, y0 = generate_sde(sde, seed=config["seed"], **config["generate_args"])

    # save train val test splits. full ts. we can compute subseq on the fly.
    val_size = config["val_size"]
    test_size = config["test_size"]

    # save_dir = Path(config["save_dir"])
    if config["save_dir"] is None:
        print("CONFIG IS NONE", config["save_dir"])
        config["save_dir"] = Path(args.config_path).parent
    save_dir = config["save_dir"]

    print("save_dir", save_dir)

    os.makedirs(save_dir, exist_ok=True)

    n = len(data)
    n_val = int(n * val_size)
    n_test = int(n * test_size)

    train_data = data[: -n_val - n_test]
    val_data = data[-n_val - n_test : -n_test]
    test_data = data[-n_test:]

    np.save(save_dir / "train_data.npy", train_data.numpy())
    np.save(save_dir / "val_data.npy", val_data.numpy())
    np.save(save_dir / "test_data.npy", test_data.numpy())

    # plot a trajectory to check that its not nans
    fig, axs = plt.subplots(1, 2, figsize=(15, 3))

    fig.delaxes(axs[0])
    ax3d = fig.add_subplot(
        1, 2, 1, projection="3d"
    )  # same position (1 row, 2 cols, index 1)
    ax3d.plot(*data[0].T, alpha=0.8)

    axs[1].plot(data[0])
    axs[1].set_xlim(0, 1000)
    plt.tight_layout()
    plt.savefig(save_dir / "sde_trajectory.png")


#
def generate_sde(
    sde,
    n_samples_per_system=10,
    burn_in=200,
    t_size=10000,
    t_end=12,
    seed=1234,
    z_score=True,
    dt=1e-4,
    brownian_size=3,
    device="cpu",
    adaptive=False,
):

    # print(seed)
    t_size += burn_in
    torch.manual_seed(seed)

    sde = sde.to(device)
    # y0 = torch.randn(n_samples_per_system, sde.dynamics.state_size) # random initial conditions: (b, n)
    y0 = torch.randn(
        n_samples_per_system, sde.dynamics.state_size, device=device
    )  # random initial conditions: (b, n)
    ts = torch.linspace(0, t_end, t_size, device=device)  # time index, (t, )
    bm = torchsde.BrownianInterval(
        ts[0],
        t_end,
        entropy=seed,
        size=(n_samples_per_system, sde.dynamics.state_size),
        device=device,
    )
    with torch.no_grad():
        ys = (
            torchsde.sdeint(
                sde=sde,
                y0=y0,
                ts=ts,
                # method="euler",
                bm=bm,
                # dt=1e-5,
                adaptive=False,
                dt_min=1e-5,
                # atol=1e-3,
                # rtol=1e-3,
                # method="euler",
            )
            .permute(1, 0, 2)
            .data.cpu()
        )  # (b, t, n)

        # dt=dt,
        # dt=1e-3,
        # device="cuda:0"

    # ys = torch.randn(n_samples_per_system, t_size, sde.dynamics.state_size)
    ys = ys[:, burn_in:]  # remove burn-in period
    if z_score:  # Normalize each system's trajectories
        ys = (ys - ys.mean((0, 1))) / ys.std()
    data = ys
    return data, y0


if __name__ == "__main__":

    parser = get_parser()
    args = parser.parse_args()

    generate_fullts(args)
    print(f"Generated data successfully from config {args.config_path}.")
