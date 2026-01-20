import torch
import torch.nn as nn
import torch.optim as optim  # <--- Needed to create the optimizer
from torch.utils.data import DataLoader, TensorDataset
from data.load_mnist_binary import load_mnist_binary_digit
from models.simple_nn import SimpleNN
from utils.training import train
from utils.imbalance import apply_smote
from utils.evaluation import evaluate
from utils.plotting import plot_confusion

def run():
    print("loading MNIST data...")
    X_train, X_test, y_train, y_test = load_mnist_binary_digit(minority_digit=0, minority_ratio=0.02)
    
    # --- SMOTE ---
    X_train, y_train = apply_smote(X_train, y_train)

    # Wrap in PyTorch Dataset
    train_dataset = TensorDataset(torch.tensor(X_train).float(), torch.tensor(y_train).long())
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    # Setup Device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Training MNIST with SMOTE...")
    # 1. FIX: Initialize Model correctly (only 1 argument)
    model = SimpleNN(784).to(device) 

    # 2. FIX: Define Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 3. FIX: Run the training loop (train function only does 1 epoch)
    epochs = 5
    for epoch in range(epochs):
        # Pass all 5 arguments that utils/training.py expects
        train(model, train_loader, criterion, optimizer, device)

    # Evaluation
    results = evaluate(model, X_test, y_test, device)
    print(f"SMOTE Results -> ROC-AUC: {results['roc_auc']:.4f}")

    # Plot
    plot_confusion(results["y_true"], results["y_pred"], ["Non-0", "0"], 
                  "MNIST SMOTE", "mnist_smote_conf.png")

if __name__ == "__main__":
    run()