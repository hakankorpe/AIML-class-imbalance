from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from utils.imbalance import create_imbalance

def load_synthetic(minority_ratio=0.1):
    X, y = make_classification(
        n_samples=5000,
        n_features=20,
        n_informative=3,
        n_redundant=1,
        weights=[0.9, 0.1],
        random_state=42
    )

    X_imb, y_imb = create_imbalance(X, y, minority_class=1, minority_ratio=minority_ratio)

    return train_test_split(X_imb, y_imb, test_size=0.2, random_state=42)
