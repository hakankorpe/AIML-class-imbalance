# utils/plotting.py
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    RocCurveDisplay,
    PrecisionRecallDisplay,
)

FIG_DIR = "figures"


def _ensure_dir():
    os.makedirs(FIG_DIR, exist_ok=True)


def plot_confusion(y_true, y_pred, labels, title, filename, normalize=True):
    """Save a confusion matrix as PNG."""
    _ensure_dir()

    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(4, 4), dpi=300)
    im = ax.imshow(cm)

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                f"{cm[i, j]:.2f}" if normalize else int(cm[i, j]),
                ha="center",
                va="center",
            )

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    out_path = os.path.join(FIG_DIR, filename)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved confusion matrix to {out_path}")


def plot_roc_pr(y_true, y_score, title_prefix, filename, pos_label=1):
    """Save ROC and PR curves (one model) as side-by-side PNG."""
    _ensure_dir()

    fig, axes = plt.subplots(1, 2, figsize=(8, 4), dpi=300)

    RocCurveDisplay.from_predictions(
        y_true, y_score, pos_label=pos_label, ax=axes[0]
    )
    axes[0].set_title(f"{title_prefix} – ROC")

    PrecisionRecallDisplay.from_predictions(
        y_true, y_score, pos_label=pos_label, ax=axes[1]
    )
    axes[1].set_title(f"{title_prefix} – PR")

    plt.tight_layout()
    out_path = os.path.join(FIG_DIR, filename)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved ROC/PR curves to {out_path}")
