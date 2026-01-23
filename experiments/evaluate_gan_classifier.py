import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from data.load_mnist_binary import load_mnist_binary_digit
from models.cgan import Generator

def evaluate_gan():
    print("--- Generating GAN Confusion Matrix ---")
    
    # 1. Load Real Data (Digit 0 vs Rest, 2% Imbalance)
    print("Loading Real Data...")
    X_train, X_test, y_train, y_test = load_mnist_binary_digit(minority_digit=0, minority_ratio=0.02)
    
    # 2. Load the GAN
    device = "cuda" if torch.cuda.is_available() else "cpu"
    z_dim = 100
    generator = Generator(z_dim, num_classes=2).to(device)
    
    model_path = "cgan_generator_0.pth" 
    if not os.path.exists(model_path):
        # Check inside experiments if not in root
        model_path = "experiments/cgan_generator_0.pth"
        
    if os.path.exists(model_path):
        print(f"Loading GAN from {model_path}...")
        generator.load_state_dict(torch.load(model_path, map_location=device))
        generator.eval()
    else:
        print(f"ERROR: Could not find {model_path}. Did you run training?")
        return

    # 3. Generate Synthetic Data
    # We want to balance the dataset. 
    # Current minority count is approx 100. Majority is ~5000.
    # We need to generate ~4900 synthetic samples.
    n_minority = np.sum(y_train == 1)
    n_majority = np.sum(y_train == 0)
    n_to_generate = n_majority - n_minority
    
    print(f"Generating {n_to_generate} synthetic samples to balance classes...")
    
    with torch.no_grad():
        z = torch.randn(n_to_generate, z_dim).to(device)
        labels = torch.ones(n_to_generate).long().to(device) # Force Class 1 (Digit 0)
        generated_data = generator(z, labels).cpu().view(n_to_generate, 784).numpy()
        
    # Scale from [-1, 1] back to [0, 1] for the classifier
    generated_data = (generated_data + 1) / 2.0
    
    # 4. Combine Real + Synthetic
    X_balanced = np.vstack((X_train, generated_data))
    y_balanced = np.hstack((y_train, np.ones(n_to_generate)))
    
    print(f"New Training Shape: {X_balanced.shape}")

    # 5. Train Classifier
    print("Training Classifier on GAN-Augmented Data...")
    clf = LogisticRegression(solver='liblinear', random_state=42)
    clf.fit(X_balanced, y_balanced)
    
    # 6. Evaluate
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]
    
    auc = roc_auc_score(y_test, y_prob)
    print(f"GAN Classifier ROC-AUC: {auc:.4f}")
    
    cm = confusion_matrix(y_test, y_pred, normalize='true')

    # 7. Plot and Save
    plt.figure(figsize=(5, 4), dpi=150)
    sns.heatmap(cm, annot=True, fmt='.2f', cmap='viridis',
                xticklabels=['Non-0', '0'], yticklabels=['Non-0', '0'])
    plt.title('MNIST GAN (Augmented)')
    plt.ylabel('True')
    plt.xlabel('Predicted')

    output_path = "figures/mnist_gan_conf.png"
    if not os.path.exists("figures"):
        os.makedirs("figures")
        
    plt.savefig(output_path, bbox_inches='tight')
    print(f"Saved confusion matrix to {output_path}")

if __name__ == "__main__":
    evaluate_gan()