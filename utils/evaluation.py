# utils/evaluation.py
import numpy as np
import torch
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    average_precision_score,
)


def evaluate(model, X_test, y_test, device, batch_size: int = 256):
    model.eval()

    X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_t = torch.tensor(y_test, dtype=torch.long).to(device)

    all_logits = []
    all_targets = []

    with torch.no_grad():
        for start in range(0, len(X_test_t), batch_size):
            end = start + batch_size
            xb = X_test_t[start:end]
            yb = y_test_t[start:end]

            logits = model(xb)
            all_logits.append(logits.cpu())
            all_targets.append(yb.cpu())

    logits = torch.cat(all_logits, dim=0)
    targets = torch.cat(all_targets, dim=0)

    # predictions and scores
    probs = torch.softmax(logits, dim=1)
    y_score = probs[:, 1].numpy()
    y_pred = probs.argmax(dim=1).numpy()
    y_true = targets.numpy()

    report = classification_report(y_true, y_pred, digits=2)
    roc_auc = roc_auc_score(y_true, y_score)
    pr_auc = average_precision_score(y_true, y_score)

    return {
        "report": report,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "y_true": y_true,
        "y_pred": y_pred,
        "y_score": y_score,
    }
