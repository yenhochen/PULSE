import os
import random

import torch
import numpy as np
import numpy.random as npr

from einops import rearrange


def set_seed(seed):
    """Set random seeds for Python, NumPy, and PyTorch (CPU + CUDA)."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def count_parameters(module, requires_grad=True):
    if requires_grad:
        return sum(p.numel() for p in module.parameters() if p.requires_grad)
    return sum(p.numel() for p in module.parameters())


def standardize_window(x, axes=(1, 2)):
    """Standardize x along given axes. Returns normalized x, mean, std."""
    if isinstance(x, torch.Tensor):
        mu = torch.mean(x, axes, keepdims=True)
        sd = torch.std(x, axes, keepdims=True)

        return (x - mu) / sd, mu.squeeze(), sd.squeeze()

    if isinstance(x, np.ndarray):
        mu = np.mean(x, axes, keepdims=True)
        sd = np.std(x, axes, keepdims=True)

        return (x - mu) / sd, mu.squeeze(), sd.squeeze()


def unstandardize_window(x, mu, sd):
    """Reverse standardize_window; x/mu/sd shapes (batch, time, channels) or broadcastable."""
    sd = rearrange(sd, "b -> b 1 1")
    mu = rearrange(mu, "b -> b 1 1")
    return x * sd + mu


def split_into_consecutive_sublists(arr):
    """
    given a numpy array, split it into a list of consecutive sublists
    arr: (n,) array.
    """
    if len(arr) == 0:
        return []

    # Find where the difference between consecutive elements is not 1
    breaks = np.where(np.diff(arr) != 1)[0]

    # Split the array based on the breaks
    sublists = np.split(arr, breaks + 1)
    return [sublist for sublist in sublists]


def get_true_rolled(x, start_ix):
    """
    x: (n, t, c) tensor
    start_ix: (n,) tensor
    """
    roll_shifts = tuple(-i.item() - 1 for i in start_ix)
    x_roll = torch.stack([torch.roll(i, s, dims=0) for i, s in zip(x, roll_shifts)])
    return x_roll


def get_roll_mask(x, start_ix):
    """
    x: (n, t, c) tensor
    start_ix: (n,) tensor
    """
    mask = torch.ones_like(x)
    for k, i in enumerate(start_ix):
        mask[k, -i - 1 :] = 0
    return mask


def get_pred_true(batch, out, start_ix=None, sample_init=False):
    """Slice prediction and target windows for reconstruction loss.

    Args:
        batch: (batch, time, channels) input window
        out: (batch, time, channels) model output
        start_ix: (batch,) start indices when sample_init is True
        sample_init: whether init position was randomly sampled
    Returns:
        true, pred tensors aligned for MSE loss
    """
    if sample_init:
        # get true values
        batch_roll = get_true_rolled(batch, start_ix)
        mask = get_roll_mask(batch_roll, start_ix)
        mask_ix = mask.sum((0, 2)) != 0
        m = mask[:, mask_ix]
        pred, true = out * m, batch_roll[:, mask_ix] * m

    else:
        pred, true = out, batch[:, 1:]

    return true, pred


def shift_and_mask(x, start_ix):
    """
    x: b, t, c
    start_ix: b,

    shift each batch index by start_ix amount then mask values on the tail end

    outputs: b, t, c but masked and shifted
    """

    x_roll = get_true_rolled(
        x, start_ix
    )  # shift time varying variables to match the start_ix
    mask = get_roll_mask(x_roll, start_ix)
    mask_ix = mask.sum((0, 2)) != 0
    m = mask[:, mask_ix]
    return x_roll[:, mask_ix] * m, m
