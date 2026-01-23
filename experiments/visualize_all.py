import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import argparse
from imblearn.over_sampling import SMOTE
from torchvision import transforms
from PIL import Image

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from data.load_mnist_binary import load_mnist_binary_digit
from models.cgan import Generator

def generate_comparison(digit):
    print(f"Generating Grand Comparison for Digit {digit}...")
    
    # 1. Load Data
    X_train, _, y_train, _ = load_mnist_binary_digit(minority_digit=digit, minority_ratio=0.02)
    
    # --- A. REAL SAMPLES ---
    X_real = X_train[y_train == 1]
    idx = np.random.choice(len(X_real), 5, replace=False)
    real_imgs = X_real[idx].reshape(-1, 28, 28)

    # --- B. AUGMENTATION ---
    # Take the first real image and rotate it 5 times
    base_img = Image.fromarray((X_real[0].reshape(28,28) * 255).astype(np.uint8))
    aug_transform = transforms.RandomRotation(20)
    aug_imgs = []
    for _ in range(5):
        aug_imgs.append(np.array(aug_transform(base_img)) / 255.0)
    aug_imgs = np.array(aug_imgs)

    # --- C. SMOTE ---
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    # Get only synthetic samples (from the end)
    X_syn = X_resampled[X_train.shape[0]:]
    y_syn = y_resampled[X_train.shape[0]:]
    X_syn_min = X_syn[y_syn == 1]
    if len(X_syn_min) > 0:
        idx = np.random.choice(len(X_syn_min), 5, replace=False)
        smote_imgs = X_syn_min[idx].reshape(-1, 28, 28)
    else:
        smote_imgs = np.zeros((5, 28, 28))

    # --- D. GAN ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    generator = Generator(100, num_classes=2).to(device)
    model_path = f"cgan_generator_{digit}.pth"
    
    gan_imgs = np.zeros((5, 28, 28))
    if os.path.exists(model_path):
        generator.load_state_dict(torch.load(model_path, map_location=device))
        generator.eval()
        with torch.no_grad():
            noise = torch.randn(5, 100).to(device)
            labels = torch.ones(5).long().to(device)
            out = generator(noise, labels).cpu().view(-1, 28, 28).numpy()
            gan_imgs = (out + 1) / 2.0
    else:
        print(f"Warning: {model_path} not found.")

    # --- PLOT ---
    fig, axes = plt.subplots(4, 5, figsize=(8, 6))
    rows = [real_imgs, aug_imgs, smote_imgs, gan_imgs]
    titles = ["Real Data", "Augmentation (Rotated)", "SMOTE (Interpolated)", "GAN (Generated)"]
    
    for row_idx, (data, title) in enumerate(zip(rows, titles)):
        for col_idx in range(5):
            ax = axes[row_idx, col_idx]
            ax.imshow(data[col_idx], cmap='gray')
            ax.axis('off')
            if col_idx == 2: # Center title
                ax.set_title(title, fontsize=10)

    plt.suptitle(f"Visualizing Data Generation Methods (Digit {digit})", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"figures/grand_comparison_{digit}.png")
    print(f"Saved to figures/grand_comparison_{digit}.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--digit", type=int, default=0)
    args = parser.parse_args()
    generate_comparison(args.digit)