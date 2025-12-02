from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from utils.imbalance import create_imbalance

def load_data(minority_ratio=0.05):
    data = load_breast_cancer()
    X, y = data.data, data.target
    
    X_imb, y_imb = create_imbalance(X, y, minority_class=1, minority_ratio=minority_ratio)
    return train_test_split(X_imb, y_imb, test_size=0.2)