"""Logging helpers for pretrain and evaluation runs."""

import logging
import os

import numpy as np

from utils.common import count_parameters
from utils.constants import LOG_BASIC_KEYS

_DEFAULT_LOGGER_NAME = "pulse"


def get_logger(name=_DEFAULT_LOGGER_NAME, log_file=None):
    """Return a module logger with optional file output.

    Handlers are added once per logger name / log file path.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    has_stream = any(isinstance(h, logging.StreamHandler) for h in logger.handlers)
    has_file = (
        any(
            isinstance(h, logging.FileHandler)
            and h.baseFilename == os.path.abspath(log_file)
            for h in logger.handlers
        )
        if log_file
        else False
    )

    formatter = logging.Formatter("%(asctime)s - %(message)s")

    if not has_stream:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    if log_file and not has_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


logger = get_logger()


def log_basic_info(config):
    """Log model type, save dir, seed, and data path from merged config."""
    rjust = 10
    for key in LOG_BASIC_KEYS:
        logger.info(f"{key.rjust(rjust)}:\t{config[key]}")
    logger.info(f"{'data_path'.rjust(rjust)}:\t{config['data_args']['path']}")
    logger.info("")


def log_parameters(trainer):
    """Log encoder and total trainable parameter counts."""
    components = [("Encoder", trainer.encoder), ("Total Parameters", trainer)]
    logger.info("Parameter Count:")
    for name, module in components:
        logger.info(f"\t{name.rjust(20)}:\t{count_parameters(module)}")
    logger.info("")


def log_scores(scores, name):
    """Log a flat dict of numeric scores."""
    logger.info("")
    logger.info(f"{name} Scores:")
    for key, value in scores.items():
        if isinstance(value, (int, float, np.floating, np.integer)):
            logger.info(f"\t{key}:\t{value:.4f}")
        else:
            logger.info(f"\t{key}:\t{value}")
    logger.info("")


def _report_column(report, score_name):
    return {k: v[score_name] for k, v in report.items() if isinstance(v, dict)}


def log_class_scores(results):
    """Log linear-probe classification, per-class F1, and per-class accuracy."""
    log_scores({k: results[k] for k in ("acc", "auroc", "auprc")}, "Classification")

    score_name = "f1-score"
    log_scores(_report_column(results["report"], score_name), f"Classification {score_name}")

    class_acc = {i: acc for i, acc in enumerate(results["cm"].diagonal())}
    log_scores(class_acc, "Class Acc")


def log_semisupervised(results):
    """Log mean accuracy across semi-supervised label fractions."""
    fractions = list(results[0].keys())
    logger.info("")
    logger.info("evaluating semisupervised classification")
    for fraction in fractions:
        acc = np.mean([run[fraction]["acc"] for run in results])
        logger.info(f"\t{fraction}:\t{acc}")
    logger.info("")
