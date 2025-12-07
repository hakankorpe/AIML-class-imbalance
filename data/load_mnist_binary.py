from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
from utils.imbalance import create_imbalance
import numpy as np

def load_mnist(minority_ratio=0.02):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(-1))
    ])

    mnist = datasets.MNIST(
        root="./", train=True, download=True, transform=transform
    )

    X = mnist.data.numpy().reshape(-1, 28 * 28) / 255.0
    y = (mnist.targets.numpy() == 0).astype(int)  # "0" becomes minority class

    X_imb, y_imb = create_imbalance(X, y, minority_class=1, minority_ratio=minority_ratio)

    return train_test_split(X_imb, y_imb, test_size=0.2, random_state=42)



def load_mnist_binary_digit(minority_digit=0, minority_ratio=0.02):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(-1))
    ])

    mnist = datasets.MNIST(root="./", train=True, download=True, transform=transform)

    X = mnist.data.numpy().reshape(-1, 28*28) / 255.0
    y = (mnist.targets.numpy() == minority_digit).astype(int)

    X_imb, y_imb = create_imbalance(
        X, y, minority_class=1, minority_ratio=minority_ratio
    )

    return train_test_split(X_imb, y_imb, test_size=0.2, random_state=42)

