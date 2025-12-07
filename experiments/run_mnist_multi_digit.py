import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.utils.class_weight import compute_class_weight

from data.load_mnist_binary import load_mnist_binary_digit
from models.simple_nn import SimpleNN
from utils.training import train
from utils.evaluation import evaluate
from utils.plotting import plot_pr_auc_comparison 

def run_digit(minority_digit):
    X_train, X_test, y_train, y_test = load_mnist_binary_digit(minority_digit=minority_digit)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SimpleNN(X_train.shape[1]).to(device)

    weights = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
    weights = torch.tensor(weights, dtype=torch.float).to(device)

    criterion = torch.nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    loader = DataLoader(
        TensorDataset(torch.tensor(X_train, dtype=torch.float),
                      torch.tensor(y_train, dtype=torch.long)),
        batch_size=64,
        shuffle=True
    )

    for epoch in range(5):
        train(model, loader, criterion, optimizer, device)

    results = evaluate(model, X_test, y_test, device)

    return {
        "digit": minority_digit,
        "roc_auc": results["roc_auc"],
        "pr_auc": results["pr_auc"],
        "report": results["report"]
    }


def run():
    digits = [0, 1, 8, 9]
    all_results = []

    for d in digits:
        print(f"\n--- Minority class: digit {d} ---")
        res = run_digit(d)
        print(res["report"])
        print("ROC-AUC:", res["roc_auc"])
        print("PR-AUC:", res["pr_auc"])
        all_results.append(res)

    return all_results


def run():
    digits = [0, 1, 8, 9]
    all_results = []

    for d in digits:
        print(f"\n--- Minority class: digit {d} ---")
        res = run_digit(d)
        print(res["report"])
        print("ROC-AUC:", res["roc_auc"])
        print("PR-AUC:", res["pr_auc"])
        all_results.append(res)

    plot_pr_auc_comparison(all_results, filename="mnist_pr_auc_digits.png")

    return all_results
