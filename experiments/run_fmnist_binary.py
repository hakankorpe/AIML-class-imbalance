import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.utils.class_weight import compute_class_weight

from data.load_fmnist_binary import load_fmnist
from models.simple_nn import SimpleNN
from utils.training import train
from utils.evaluation import evaluate
from utils.plotting import plot_confusion, plot_roc_pr


def run():
    # --- correct loader ---
    X_train, X_test, y_train, y_test = load_fmnist(minority_ratio=0.02)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SimpleNN(X_train.shape[1]).to(device)

    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_train),
        y=y_train
    )
    weights = torch.tensor(weights, dtype=torch.float).to(device)

    criterion = torch.nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    loader = DataLoader(
        TensorDataset(
            torch.tensor(X_train, dtype=torch.float),
            torch.tensor(y_train, dtype=torch.long)
        ),
        batch_size=64,
        shuffle=True
    )

    for epoch in range(5):
        train(model, loader, criterion, optimizer, device)

    results = evaluate(model, X_test, y_test, device)
    print(results["report"])
    print("ROC-AUC:", results["roc_auc"])
    print("PR-AUC:", results["pr_auc"])

    # --- get outputs for plotting ---
    y_true = results["y_true"]
    y_pred = results["y_pred"]
    y_score = results["y_score"]

    # --- Confusion Matrix ---
    plot_confusion(
        y_true=y_true,
        y_pred=y_pred,
        labels=["Other", "T-shirt"],
        title="Fashion-MNIST Binary – Class Weights",
        filename="fmnist_confusion.png",
    )

    # --- ROC + PR curves ---
    plot_roc_pr(
        y_true=y_true,
        y_score=y_score,
        title_prefix="Fashion-MNIST Binary",
        filename="fmnist_roc_pr.png",
    )
