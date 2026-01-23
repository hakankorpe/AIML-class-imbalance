import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from imblearn.over_sampling import SMOTE

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from data.load_mnist_binary import load_mnist_binary_digit

def visualize_smote_samples():
    print("Loading MNIST data...")
    # Load the same 2% imbalance data
    X_train, _, y_train, _ = load_mnist_binary_digit(minority_digit=0, minority_ratio=0.02)

    print(f"Original shape: {X_train.shape}, Minority count: {np.sum(y_train == 1)}")

    # --- Apply SMOTE ---
    print("Applying SMOTE...")
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

    print(f"Resampled shape: {X_resampled.shape}, Minority count: {np.sum(y_resampled == 1)}")

    # --- Identify Synthetic Samples ---
    # The synthetic samples are appended to the end of the original data.
    n_original = X_train.shape[0]
    X_synthetic = X_resampled[n_original:]
    y_synthetic = y_resampled[n_original:]

    # Filter for the minority class (digit 0, which is labeled as 1 here)
    minority_indices = np.where(y_synthetic == 1)[0]
    X_synthetic_minority = X_synthetic[minority_indices]

    print(f"Number of synthetic 'zeros' generated: {X_synthetic_minority.shape[0]}")

    if X_synthetic_minority.shape[0] == 0:
        print("Error: No synthetic minority samples found.")
        return

    # --- Plot ---
    num_samples_to_plot = 16
    # Get a random selection or the first N samples
    indices = np.random.choice(X_synthetic_minority.shape[0], num_samples_to_plot, replace=False)
    samples_to_plot = X_synthetic_minority[indices]

    fig, axes = plt.subplots(2, 8, figsize=(12, 3))
    fig.suptitle("Generated Synthetic 'Zeros' (SMOTE)", fontsize=16)

    for i, ax in enumerate(axes.flatten()):
        if i < len(samples_to_plot):
            # Reshape from 1D vector back to 2D image
            img = samples_to_plot[i].reshape(28, 28)
            ax.imshow(img, cmap='gray')
            ax.axis('off')
        else:
            ax.axis('off') # Hide unused subplots

    output_path = "figures/smote_samples.png"
    plt.savefig(output_path)
    print(f"Saved SMOTE samples to {output_path}")
    plt.show()

if __name__ == "__main__":
    visualize_smote_samples()