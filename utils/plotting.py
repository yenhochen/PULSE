"""Visualization helpers for embeddings and PULSE reconstructions."""

import numpy as np
import matplotlib.pyplot as plt
from cuml import TSNE
from matplotlib.lines import Line2D
from sklearn.metrics import r2_score


def compute_tsne(x, tsne_kwargs=None):
    """Project embeddings to 2D with cuML t-SNE.

    Args:
        x: (n_samples, n_features)
        tsne_kwargs: forwarded to cuml.TSNE
    Returns:
        fitted TSNE model, (n_samples, 2) embedding
    """
    if tsne_kwargs is None:
        tsne_kwargs = {"perplexity": 30, "random_state": 0}
    tsne = TSNE(**tsne_kwargs)
    emb = tsne.fit_transform(x)
    return tsne, emb


def plot_emb_ax(emb, labels, label_names, ax, alpha=0.1, show_legend=True):
    """Scatter 2D embedding on ax, colored by class label."""
    cmap = plt.get_cmap("tab10")
    colors = [cmap(i) for i in range(10)]
    unique_labels = np.unique(labels)

    ax.set_aspect(1)
    for label in unique_labels:
        ax.scatter(
            *emb[labels == label].T,
            alpha=alpha,
            c=labels[labels == label],
            cmap="tab10",
            vmin=0,
            vmax=10,
        )

    if show_legend:
        handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=color,
                markersize=10,
                label=label_names[label],
            )
            for color, label in zip(colors, unique_labels)
        ]
        ax.legend(handles=handles, bbox_to_anchor=(1, 0.75))


def plot_reconstruction(config, true, pred, labels, save_path, save=True, show=False):
    """Plot true vs predicted windows and scatter R² for random samples."""
    n_show = 10
    subseq_size = config["data_args"]["subseq_size"]

    rng = np.random.default_rng(1234)
    sample_ixs = rng.permutation(len(true))[:n_show]

    fig, axs = plt.subplots(n_show, 3, figsize=(10, 2 * n_show), dpi=150)

    for row, ix in enumerate(sample_ixs):
        t = true[ix][:subseq_size]
        p = pred[ix][:subseq_size]
        y = labels[ix]

        axs[row, 0].plot(t)
        axs[row, 0].set_title(f"ix: {ix} y: {y}. True")

        axs[row, 1].plot(p)
        axs[row, 1].set_title("Pred")

        ylim = _shared_ylim(axs[row])
        axs[row, 0].set_ylim(ylim)
        axs[row, 1].set_ylim(ylim)

        r2 = r2_score(t.flatten(), p.flatten())
        axs[row, 2].scatter(t.flatten(), p.flatten(), alpha=0.1)
        axs[row, 2].plot([-2, 2], [-2, 2], "r--")
        axs[row, 2].set_aspect(1)
        axs[row, 2].set_title(f"r2: {r2:.3f}")

    plt.tight_layout()

    if save:
        plt.savefig(save_path)
    if show:
        plt.show()
    plt.close(fig)


def _shared_ylim(ax_row):
    """Union y-limits across a row of axes."""
    ymins, ymaxs = zip(*(ax.get_ylim() for ax in ax_row))
    return min(ymins), max(ymaxs)
