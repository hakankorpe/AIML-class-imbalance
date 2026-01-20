import numpy as np
from imblearn.over_sampling import SMOTE

def create_imbalance(X, y, minority_class=1, minority_ratio=0.1):
    X = np.array(X)
    y = np.array(y)

    minority_idx = np.where(y == minority_class)[0]
    majority_idx = np.where(y != minority_class)[0]

    np.random.shuffle(minority_idx)
    keep_minority = int(len(minority_idx) * minority_ratio)
    minority_idx = minority_idx[:keep_minority]

    new_idx = np.concatenate([majority_idx, minority_idx])
    np.random.shuffle(new_idx)

    return X[new_idx], y[new_idx]


def apply_smote(X, y):
    """
    Applies SMOTE to numeric data (Phase 2).
    """
    print(f"Applying SMOTE... Original shape: {X.shape}")
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X, y)
    print(f"SMOTE applied. New shape: {X_res.shape}")
    return X_res, y_res