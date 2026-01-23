import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from torchvision import transforms
from PIL import Image

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from data.load_mnist_binary import load_mnist_binary_digit

def visualize_augmentation(digit=0):
    print(f"--- Visualizing Augmentation for Digit {digit} ---")
    
    # 1. Load Real Data
    X_train, _, y_train, _ = load_mnist_binary_digit(minority_digit=digit, minority_ratio=0.02)
    # Get just one real sample to augment
    real_sample = X_train[y_train == 1][0].reshape(28, 28)
    
    # Convert to PIL for torchvision transforms
    # MNIST is 0-1 float, convert to 0-255 uint8
    img_pil = Image.fromarray((real_sample * 255).astype(np.uint8))

    # 2. Define Augmentations (Same as you likely used in training)
    # Rotations +/- 15 degrees, slight shifts
    augmenter = transforms.Compose([
        transforms.RandomRotation(degrees=20),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    ])

    # 3. Generate Variations of the SAME image
    fig, axes = plt.subplots(1, 8, figsize=(12, 2))
    
    # Plot original first
    axes[0].imshow(real_sample, cmap='gray')
    axes[0].set_title("Original")
    axes[0].axis('off')
    
    # Plot 7 augmented versions
    for i in range(1, 8):
        # Apply random transform
        aug_img = augmenter(img_pil)
        axes[i].imshow(np.array(aug_img), cmap='gray')
        axes[i].set_title("Augmented")
        axes[i].axis('off')
        
    plt.suptitle(f"Geometric Augmentation (Digit {digit})", fontsize=16)
    filename = f"figures/aug_samples_{digit}.png"
    plt.savefig(filename)
    print(f"Saved visualization to {filename}")

if __name__ == "__main__":
    # Can change this manually or use argparse here too
    visualize_augmentation(digit=0)
    visualize_augmentation(digit=8)