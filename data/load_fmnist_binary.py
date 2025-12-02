from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
from utils.imbalance import create_imbalance
import numpy as np

def load_fmnist(minority_ratio=0.02):
    """
    Fashion-MNIST binary classification:
    T-shirt/top (class 0) vs all other classes.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(-1))
    ])

    fmnist = datasets.FashionMNIST(
        root="./", train=True, download=True, transform=transform
    )

    # Use "T-shirt/top" (label 0) as the positive (minority) class
    X = fmnist.data.numpy().reshape(-1, 28 * 28) / 255.0
    y = (fmnist.targets.numpy() == 0).astype(int)

    # Create artificial imbalance (e.g. 2% positives)
    X_imb, y_imb = create_imbalance(X, y, minority_class=1, minority_ratio=minority_ratio)

    return train_test_split(X_imb, y_imb, test_size=0.2, random_state=42)
