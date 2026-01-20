import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import sys
from torch.utils.data import DataLoader, TensorDataset

# Fix path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from data.load_mnist_binary import load_mnist_binary_digit
from models.simple_nn import SimpleNN
from models.cgan import Generator
from utils.training import train
from utils.evaluation import evaluate
from utils.plotting import plot_confusion

def generate_synthetic_data(n_samples, device):
    """Generates fake zeros using the trained GAN."""
    z_dim = 100
    generator = Generator(z_dim, num_classes=2).to(device)
    
    # Load the trained model
    model_path = "cgan_generator.pth"
    if not os.path.exists(model_path):
        model_path = "experiments/cgan_generator.pth"
        
    try:
        generator.load_state_dict(torch.load(model_path, map_location=device))
    except FileNotFoundError:
        print("ERROR: Run 'experiments/train_gan.py' first!")
        return None, None

    generator.eval()
    with torch.no_grad():
        # Generate Class '1' (Minority/Zero)
        noise = torch.randn(n_samples, z_dim).to(device)
        labels = torch.ones(n_samples).long().to(device) 
        
        fake_imgs = generator(noise, labels).cpu().numpy()
        
        # Rescale [-1, 1] back to [0, 1]
        fake_imgs = (fake_imgs + 1) / 2.0
        
    return fake_imgs, np.ones(n_samples)

def run():
    print("loading MNIST data...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Load Data
    X_train, X_test, y_train, y_test = load_mnist_binary_digit(minority_digit=0, minority_ratio=0.02)
    
    # 2. GAN Balancing (Phase 3)
    n_maj = np.sum(y_train == 0)
    n_min = np.sum(y_train == 1)
    n_needed = n_maj - n_min
    
    print(f"Generating {n_needed} fake samples using GAN...")
    X_fake, y_fake = generate_synthetic_data(n_needed, device)
    
    if X_fake is not None:
        X_train = np.concatenate((X_train, X_fake), axis=0)
        y_train = np.concatenate((y_train, y_fake), axis=0)
    
    # 3. Create Loader
    dataset = TensorDataset(torch.tensor(X_train).float(), torch.tensor(y_train).long())
    train_loader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    print("Training Classifier on GAN-Augmented Data...")
    # 1. FIX: Correct model init
    model = SimpleNN(784).to(device)

    # 2. FIX: Define Optimizer and Loss
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 3. FIX: Training Loop with correct arguments
    epochs = 5
    for epoch in range(epochs):
        train(model, train_loader, criterion, optimizer, device)
        
    # Evaluate
    results = evaluate(model, X_test, y_test, device)
    print(f"GAN Results -> ROC-AUC: {results['roc_auc']:.4f}")

    plot_confusion(results["y_true"], results["y_pred"], ["Non-0", "0"], 
                  "MNIST GAN", "mnist_gan_conf.png")

if __name__ == "__main__":
    run()