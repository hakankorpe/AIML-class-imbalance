import torch
from torch.utils.data import DataLoader, TensorDataset
from imblearn.over_sampling import SMOTE

from data.load_breast_cancer import load_data
from models.simple_nn import SimpleNN
from utils.training import train
from utils.evaluation import evaluate
from utils.plotting import plot_confusion, plot_roc_pr  # NEW


def run():
    X_train, X_test, y_train, y_test = load_data(minority_ratio=0.05)

    sm = SMOTE()
    X_train, y_train = sm.fit_resample(X_train, y_train)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SimpleNN(X_train.shape[1]).to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    loader = DataLoader(
        TensorDataset(torch.tensor(X_train, dtype=torch.float),
                      torch.tensor(y_train, dtype=torch.long)),
        batch_size=32,
        shuffle=True
    )

    for epoch in range(10):
        train(model, loader, criterion, optimizer, device)

    results = evaluate(model, X_test, y_test, device)
    print(results["report"])
    print("ROC-AUC:", results["roc_auc"])
    print("PR-AUC:", results["pr_auc"])

    # ------- NEW: plots -------
    y_true = results["y_true"]
    y_pred = results["y_pred"]
    y_score = results["y_score"]

    plot_confusion(
        y_true=y_true,
        y_pred=y_pred,
        labels=["Negative", "Positive"],
        title="Breast Cancer – SMOTE",
        filename="bc_confusion_smote.png",
    )

    plot_roc_pr(
        y_true=y_true,
        y_score=y_score,
        title_prefix="Breast Cancer – SMOTE",
        filename="bc_roc_pr_smote.png",
    )
