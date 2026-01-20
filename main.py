import sys
import os

# Existing experiments
from experiments.run_breast_cancer import run as exp_breast
from experiments.run_breast_cancer_smote import run as exp_smote
from experiments.run_mnist_binary import run as exp_mnist
from experiments.run_fmnist_binary import run as exp_fmnist
from experiments.run_mnist_multi_digit import run as exp_mnist_multi

# --- NEW EXPERIMENTS FOR POSTER 2 ---
# (Make sure these files exist in your experiments folder!)
from experiments.run_mnist_smote import run as exp_mnist_smote
from experiments.run_mnist_aug import run as exp_mnist_aug
from experiments.run_mnist_gan import run as exp_mnist_gan   # <--- Added GAN

if __name__ == "__main__":
    # --- POSTER 1 EXPERIMENTS ---
    print("\n=== Running Breast Cancer (Class Weights) ===")
    exp_breast()

    print("\n=== Running Breast Cancer (SMOTE) ===")
    exp_smote()

    print("\n=== Running MNIST Binary (Class Weights) ===")
    exp_mnist()

    print("\n=== Running Fashion-MNIST Binary (Class Weights) ===")
    exp_fmnist()

    print("\n=== Running MNIST multi-digit imbalance experiment ===")
    exp_mnist_multi()

    # --- POSTER 2 EXPERIMENTS (NEW) ---
    print("\n=== Running MNIST (SMOTE - Phase 2) ===")
    exp_mnist_smote()

    print("\n=== Running MNIST (Augmentation - Phase 1) ===")
    exp_mnist_aug()

    print("\n=== Running MNIST (GAN - Phase 3) ===")
    # Check if the GAN model exists before running
    if os.path.exists("cgan_generator.pth"):
        exp_mnist_gan()
    else:
        print("[!] SKIP: GAN model not found. Run 'experiments/train_gan.py' first.")