from experiments.run_breast_cancer import run as exp_breast
from experiments.run_breast_cancer_smote import run as exp_smote
from experiments.run_mnist_binary import run as exp_mnist
from experiments.run_fmnist_binary import run as exp_fmnist


if __name__ == "__main__":
    print("\n=== Running Breast Cancer (Class Weights) ===")
    exp_breast()

    print("\n=== Running Breast Cancer (SMOTE) ===")
    exp_smote()

    print("\n=== Running MNIST Binary (Class Weights) ===")
    exp_mnist()

    print("\n=== Running Fashion-MNIST Binary (Class Weights) ===")
    exp_fmnist()