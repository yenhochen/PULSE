"""Reconstruction and regression metrics."""

import numpy as np
from sklearn.metrics import r2_score


def mse_loss(true, pred):
    """Mean squared error."""
    return ((true - pred) ** 2).mean()


def mae_loss(true, pred):
    """Mean absolute error."""
    return np.abs(true - pred).mean()


def compute_scores(true, pred):
    """Return MSE, R², and MAE for aligned true/pred arrays."""
    return {
        "MSE": mse_loss(true, pred),
        "R2": r2_score(true, pred),
        "MAE": mae_loss(true, pred),
    }
