import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from data.load_mnist_binary import load_mnist_binary_digit
from models.simple_nn import SimpleNN
from utils.training import train
from utils.evaluation import evaluate
from utils.plotting import plot_confusion
from utils.augmentation import AugmentedDataset 

def run():
    print("loading MNIST data...")
    # Load raw data
    X_train, X_test, y_train, y_test = load_mnist_binary_digit(minority_digit=0, minority_ratio=0.02)

    # Wrap in Augmentation Dataset (Phase 1)
    train_dataset = AugmentedDataset(X_train, y_train)
    
    # Test dataset is standard (No augmentation)
    test_dataset = TensorDataset(
        torch.tensor(X_test, dtype=torch.float), 
        torch.tensor(y_test, dtype=torch.long)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    # Device setup
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Training MNIST with Augmentation...")
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
    print(f"Augmentation Results -> ROC-AUC: {results['roc_auc']:.4f}")
    
    plot_confusion(results["y_true"], results["y_pred"], ["Non-0", "0"], 
                  "MNIST Augmentation", "mnist_aug_conf.png")

if __name__ == "__main__":
    run()