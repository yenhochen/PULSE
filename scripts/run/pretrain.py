"""Main experiment entry point for pretraining and downstream evaluation.

Loads a YAML config, trains the requested model, then (if eval_args.do_eval)
runs linear probe, k-means clustering, semi-supervised eval, t-SNE, and
PULSE reconstruction plots. Writes results under config save_dir.

CLI overrides (applied after config merge):
  -c / --config_path   Experiment YAML
  -s / --seed           Random seed (overrides config seed)
  -e / --epochs         Training epochs (rebar: also caps cross-attn stage 1)
  -p / --data_path      Processed data directory
  -sd / --save_dir      Output directory for checkpoints and scores.pkl
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    accuracy_score,
)
from sklearn.preprocessing import label_binarize

from utils.io import get_full_config, get_date, load_trainer, save_pickle, save_json
from utils.dataset import (
    get_data_from_config,
    get_eval_dataloader,
    get_label_names,
    SPLIT_LIST,
)
from utils.logging import (
    log_parameters,
    log_basic_info,
    log_scores,
    log_class_scores,
    log_semisupervised,
    get_logger,
)

from utils.metrics import compute_scores
from utils.evaluate import eval_classification, eval_cluster, eval_semisupervised
from utils.plotting import compute_tsne, plot_emb_ax, plot_reconstruction

from utils.common import set_seed
from utils.constants import CONTRASTIVE


def apply_epochs_override(config, epochs):
    """Override SSL epoch count; for rebar also cap cross-attn pretraining stage."""
    config.training_args.epochs = epochs
    if config.model_type == "rebar":
        config.model_args.rebarcrossattn_epochs = epochs
        config.model_args.rebarcrossattn_save_epochfreq = min(
            config.model_args.rebarcrossattn_save_epochfreq, epochs
        )


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config_path", help="config path yaml file", type=str, default=None
    )
    parser.add_argument(
        "-s",
        "--seed",
        help="set seed. this overrides the config file",
        type=str,
        default=None,
    )
    parser.add_argument(
        "-e",
        "--epochs",
        help="training epochs (overrides config; rebar cross-attn stage capped too)",
        type=int,
        default=None,
    )
    parser.add_argument(
        "-p",
        "--data_path",
        help="set data_path. this overrides the config file",
        type=str,
        default=None,
    )
    parser.add_argument(
        "-sd",
        "--save_dir",
        help="set save_dir. this overrides the config file",
        type=str,
        default=None,
    )
    return parser


def supervised_classification(y_true, y_proba, y_pred, n_classes):
    # y_score = linearprobe_classifier.predict_proba(test_repr)
    # y_pred = linearprobe_classifier.predict(test_repr)
    # acc = linearprobe_classifier.score(test_repr, test_labels)

    acc = accuracy_score(y_true, y_pred)

    report = classification_report(y_true, y_pred, output_dict=True)
    cm = confusion_matrix(y_true, y_pred, normalize="pred")

    test_labels_onehot = label_binarize(y_true, classes=np.arange(n_classes + 1))
    if n_classes + 1 == 2:
        test_labels_onehot = label_binarize(y_true, classes=np.arange(n_classes + 2))
        test_labels_onehot = test_labels_onehot[:, :2]
    auprc = average_precision_score(test_labels_onehot, y_proba)
    auroc = roc_auc_score(test_labels_onehot, y_proba)

    return {
        "acc": acc,
        "auroc": auroc,
        "auprc": auprc,
        "report": report,
        "cm": cm,
    }


def run(config):

    set_seed(config.seed)

    # generate save_dir based on date and model type
    if config["save_dir"] is None:
        data_name = [i for i in config["data_args"]["path"].split("/") if i != "data"][
            0
        ]
        config["save_dir"] = os.path.join(
            f"experiments", data_name, config["model_type"], get_date()
        )
    save_dir = Path(config["save_dir"])

    os.makedirs(config["save_dir"], exist_ok=True)

    log_basic_info(config)

    log_file = save_dir / "log.log"
    logger = get_logger(log_file=log_file)
    logger.info(f"Log file created:\t{log_file}")

    if config.model_type in CONTRASTIVE:  # contrastive must be fullts
        logger.info(
            f"Mode {config.model_type} must be fullts. setting data_type to fullts"
        )
        config.data_args.mode = "fullts"

    # data = get_data_from_config(config, config.data_args.mode)

    logger.info(f"Config saved to:\t{save_dir / 'config.yaml'}")
    logger.info(f"")

    data, info = get_data_from_config(config, config.data_args.mode, train=True)
    logger.info(f"Loading data from:\t{config.data_args.path}")
    logger.info(f"train data shape:\t{data['train_data'].shape}")
    logger.info(f"val data shape:\t{data['val_data'].shape}")
    logger.info(f"")

    trainer, is_trained = load_trainer(config, data, config.load_from_checkpoint)

    # config["data_args"]["path"] = str(config["data_args"]["path"])

    # print(config)
    log_parameters(trainer)
    logger.info("Fitting model...")

    if not is_trained:  # train if not trained
        metrics_dict = trainer.fit()

    # print(config)

    if config["eval_args"]["do_eval"]:

        trainer, is_trained = load_trainer(config, data, "checkpoint_best")

        logger.info(f"Evaluating model Fit...")

        eval_loaders, data_info = get_eval_dataloader(config)

        is_running_on_analysis_data = "analysis_args" in config["data_args"]
        # label_names = data_info.label_names if is_running_on_analysis_data else get_label_names(config)
        if is_running_on_analysis_data:
            label_names = data_info.label_names
            analysis_info = {
                "paths": [str(p) for p in data_info.paths],
                "label_names": label_names,
            }

            ainfo_path = save_dir / "analysis_info.json"
            save_json(analysis_info, ainfo_path)
            logger.info(f"Saving analysis info to:\t{ainfo_path}")
        else:
            label_names = get_label_names(config)

        # ============================== evaluate model ==============================

        eval_results = {}
        for split in SPLIT_LIST:
            results = trainer.evaluate(eval_loaders[f"{split}_loader"])
            results["embed"] = results["embed"].squeeze()
            eval_results[f"{split}_results"] = results

            if "pred" in results:  # compute reconstruction scores if available.
                scores = compute_scores(
                    results["true"].flatten(), results["pred"].flatten()
                )
                eval_results[f"{split}_scores"] = scores
                log_scores(scores, split)
            else:
                eval_results[f"{split}_scores"] = None

        logger.info("Evaluating Learned Representation...")

        if config.model_type != "supervised":
            eval_class_results = eval_classification(
                eval_results["train_results"]["embed"],
                eval_results["train_results"]["labels"],
                eval_results["test_results"]["embed"],
                eval_results["test_results"]["labels"],
            )

            eval_clust_results = eval_cluster(
                eval_results["train_results"]["embed"],
                eval_results["train_results"]["labels"],
                eval_results["test_results"]["embed"],
                eval_results["test_results"]["labels"],
                k=len(label_names),
            )
        else:
            eval_class_results = supervised_classification(
                eval_results["test_results"]["labels"],
                eval_results["test_results"]["pred_proba"],
                eval_results["test_results"]["pred_labels"],
                len(np.unique(eval_results["train_results"]["labels"])),
            )
            eval_clust_results = {}

        log_class_scores(eval_class_results)
        log_scores(eval_clust_results, "Cluster")

        if config.model_type != "supervised" and not is_running_on_analysis_data:
            eval_semisupervised_results = eval_semisupervised(
                config,
                eval_results["train_results"]["embed"],
                eval_results["train_results"]["labels"],
                eval_results["test_results"]["embed"],
                eval_results["test_results"]["labels"],
            )
            log_semisupervised(eval_semisupervised_results)
        else:
            eval_semisupervised_results = None

        # ============================== save scores ==============================

        scores = {
            "train_recon": eval_results["train_scores"],
            "val_recon": eval_results["val_scores"],
            "test_recon": eval_results["val_scores"],
            "classification": eval_class_results,
            "cluster": eval_clust_results,
            "semisupervised": eval_semisupervised_results,
        }

        # save representations
        for split in SPLIT_LIST:
            repr_path = save_dir / f"encoded_{split}.npy"
            np.save(repr_path, eval_results[f"{split}_results"]["embed"])
            logger.info(f"Saving representations to: {repr_path}")

        # save scores
        score_path = save_dir / "scores.pkl"
        save_pickle(scores, score_path)
        logger.info(f"Saving scores to {score_path}")

        # ============================== plotting ==============================

        train_repr, test_repr = (
            eval_results["train_results"]["embed"],
            eval_results["test_results"]["embed"],
        )
        train_labels, test_labels = (
            eval_results["train_results"]["labels"],
            eval_results["test_results"]["labels"],
        )

        repr = np.concatenate([train_repr, test_repr])
        labels = np.concatenate([train_labels, test_labels])

        tsne, emb = compute_tsne(
            repr,
            tsne_kwargs={
                "perplexity": 50,
                "random_state": config.seed,
                "n_neighbors": 200,
                "init": "pca",
                "learning_rate": max(len(repr) / 12 / 4, 50),
                "method": "barnes_hut",
            },  # cuML
            #  tsne_kwargs={"perplexity": 100, "random_state": 1234} # sklearn
        )
        train_emb = emb[: len(train_repr)]
        eval_emb = emb[len(train_repr) :]

        tsne_save_path = Path(config.save_dir) / "tsne.png"
        # os.path.join(config.run_dir, "tsne.png")

        subplots_kwargs = {"figsize": (8, 4), "dpi": 250}

        # plot_tsne_alpha
        fig, ax = plt.subplots(1, 2, **subplots_kwargs)
        plot_emb_ax(
            train_emb,
            train_labels,
            label_names,
            ax[0],
            alpha=config["eval_args"]["plot_tsne_alpha"],
            show_legend=False,
        )
        plot_emb_ax(
            eval_emb,
            test_labels,
            label_names,
            ax[1],
            alpha=config["eval_args"]["plot_tsne_alpha"],
        )

        xlim = ax[0].get_xlim()
        ylim = ax[0].get_ylim()

        ax[1].set_ylim(ylim)
        ax[1].set_xlim(xlim)

        ax[0].set_title("Train set")
        ax[1].set_title("Val set")
        plt.tight_layout()
        plt.savefig(tsne_save_path)

        logger.info(f"saving TSNE plot to: {tsne_save_path}")

        if config.model_type == "pulse":
            # reconstruction plots
            true, pred, labels = (
                eval_results["train_results"]["true"],
                eval_results["train_results"]["pred"],
                eval_results["train_results"]["labels"],
            )

            plot_reconstruction(
                config,
                true,
                pred,
                labels,
                Path(config.save_dir) / "recon_train.png",
                save=True,
                show=False,
            )

            true, pred, labels = (
                eval_results["val_results"]["true"],
                eval_results["val_results"]["pred"],
                eval_results["val_results"]["labels"],
            )
            plot_reconstruction(
                config,
                true,
                pred,
                labels,
                Path(config.save_dir) / "recon_val.png",
                save=True,
                show=False,
            )

            true, pred, labels = (
                eval_results["test_results"]["true"],
                eval_results["test_results"]["pred"],
                eval_results["test_results"]["labels"],
            )
            plot_reconstruction(
                config,
                true,
                pred,
                labels,
                Path(config.save_dir) / "recon_test.png",
                save=True,
                show=False,
            )


if __name__ == "__main__":

    parser = get_parser()
    args = parser.parse_args()

    config = get_full_config(args.config_path)

    if args.seed is not None:
        config["seed"] = int(args.seed)
    if args.data_path is not None:
        config["data_args"]["path"] = args.data_path
    if args.save_dir is not None:
        config["save_dir"] = args.save_dir
    if args.epochs is not None:
        apply_epochs_override(config, args.epochs)

    run(config)
