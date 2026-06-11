"""Downstream evaluation of frozen encoder representations.

Linear probe uses cuML LogisticRegression on standardized embeddings.
Semi-supervised eval subsamples 1% and 5% of training labels (5 random draws
per run, seeded by config.seed + draw index) and fits a linear probe on each.
"""

import warnings
import numpy as np
from cuml import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import label_binarize, StandardScaler
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    silhouette_score,
    davies_bouldin_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
)
from utils.logging import get_logger


logger = get_logger()


def eval_classification(
    train_repr,
    train_labels,
    val_repr,
    val_labels,
    test_repr=None,
    test_labels=None,
):
    """Fit a linear probe on train_repr and evaluate on test_repr (or val_repr)."""

    if test_repr is None:
        test_repr = val_repr
        test_labels = val_labels

    linearprobe_classifier = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            verbose=0,
        ),
    )

    linearprobe_classifier.fit(train_repr, train_labels)

    y_score = linearprobe_classifier.predict_proba(test_repr)
    y_pred = linearprobe_classifier.predict(test_repr)
    acc = linearprobe_classifier.score(test_repr, test_labels)

    report = classification_report(test_labels, y_pred, output_dict=True)
    cm = confusion_matrix(test_labels, y_pred, normalize="pred")

    test_labels_onehot = label_binarize(
        test_labels, classes=np.arange(train_labels.max() + 1)
    )
    if train_labels.max() + 1 == 2:
        test_labels_onehot = label_binarize(
            test_labels, classes=np.arange(train_labels.max() + 2)
        )
        test_labels_onehot = test_labels_onehot[:, :2]
    auprc = average_precision_score(test_labels_onehot, y_score)
    auroc = roc_auc_score(test_labels_onehot, y_score)

    return {
        "acc": acc,
        "auroc": auroc,
        "auprc": auprc,
        "report": report,
        "cm": cm,
        "classifier": linearprobe_classifier,
    }


def eval_cluster(
    val_repr,
    val_labels,
    test_repr=None,
    test_labels=None,
    k=None,
):
    """Cluster test_repr with k-means and report unsupervised metrics."""

    if test_repr is None:
        test_repr = val_repr
        test_labels = val_labels

    if k is None:
        k = len(np.unique(test_labels))

    kmeans = KMeans(n_clusters=k, random_state=10, n_init="auto").fit(test_repr)
    cluster_labels = kmeans.labels_
    s_score = silhouette_score(test_repr, cluster_labels)
    db_score = davies_bouldin_score(test_repr, cluster_labels)
    ar_score = adjusted_rand_score(cluster_labels, test_labels)
    nmi_score = normalized_mutual_info_score(cluster_labels, test_labels)

    return {"sil": s_score, "db": db_score, "ari": ar_score, "nmi": nmi_score, "k": k}


def eval_semisupervised(
    config, train_repr, train_labels, test_repr, test_labels, n_label_smooth=1
):
    """Evaluate representations with 1% and 5% label fractions (see README.md)."""
    logger.info("Running semi-supervised evaluation ... ")
    n_train_samples = len(train_repr)

    classes = np.unique(train_labels)

    label_percentage = [0.01, 0.05]
    semi_supervised_results = []
    for seed in range(config["eval_args"]["eval_n_seeds"]):

        semi_supervised_results_seed = {p: None for p in label_percentage}
        for p_train in label_percentage:

            train_ixs = np.arange(n_train_samples)
            np.random.seed(config["seed"] + seed)
            n_train_samples_p = int(p_train * n_train_samples)
            train_ixs = np.random.permutation(train_ixs)
            chosen_ixs = train_ixs[:n_train_samples_p]
            chosen_ixs_set = set(chosen_ixs)

            for u in classes:
                label_ixs = np.where(train_labels[train_ixs] == u.item())[0]
                label_ix = np.random.permutation(label_ixs)[:n_label_smooth]
                [chosen_ixs_set.add(i) for i in label_ix]

            chosen_ixs = np.array(list(chosen_ixs_set))

            train_repr_p = train_repr[chosen_ixs]
            train_labels_p = train_labels[chosen_ixs]

            u, c = np.unique(train_labels_p, return_counts=True)

            logger.info(
                f"\tSeed: {seed}\tLabel Percent: {p_train}\tLabels: {u} Count: {c}"
            )

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                eval_class_results = eval_classification(
                    train_repr_p, train_labels_p, test_repr, test_labels
                )

            semi_supervised_results_seed[p_train] = eval_class_results
        semi_supervised_results.append(semi_supervised_results_seed)

    return semi_supervised_results
