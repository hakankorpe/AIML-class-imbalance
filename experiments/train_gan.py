import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import sys
import argparse 

# Add parent directory to path to import models/data
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from data.load_mnist_binary import load_mnist_binary_digit
from models.cgan import Generator, Discriminator

# Hyperparameters
BATCH_SIZE = 64
LR = 0.0002
Z_DIM = 100
EPOCHS = 100 

def train(digit):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"--- Training GAN for Digit {digit} ---")

    # 1. Load Data (Dynamic Digit)
    # This loads ONLY the minority class for that specific digit
    X_train, _, y_train, _ = load_mnist_binary_digit(minority_digit=digit, minority_ratio=0.02)
    
    # Select only the minority samples (the ones we want to generate)
    # In binary setup, minority is class 1
    X_minority = X_train[y_train == 1]
    
    # Convert to PyTorch
    dataset = torch.tensor(X_minority).float()
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 2. Init Models
    generator = Generator(Z_DIM, num_classes=2).to(device)
    discriminator = Discriminator(num_classes=2).to(device)
    
    optimizer_G = optim.Adam(generator.parameters(), lr=LR, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=LR, betas=(0.5, 0.999))
    
    criterion = nn.BCELoss()

    # 3. Training Loop
    for epoch in range(EPOCHS):
        for i, real_imgs in enumerate(dataloader):
            batch_size = real_imgs.size(0)
            real_imgs = real_imgs.to(device)
            
            # Labels
            real_labels = torch.ones(batch_size, 1).to(device)
            fake_labels = torch.zeros(batch_size, 1).to(device)
            # Class labels for cGAN (all are class 1 "Minority")
            class_labels = torch.ones(batch_size).long().to(device) 

            # --- Train Discriminator ---
            optimizer_D.zero_grad()
            outputs = discriminator(real_imgs, class_labels)
            d_loss_real = criterion(outputs, real_labels)
            
            z = torch.randn(batch_size, Z_DIM).to(device)
            fake_imgs = generator(z, class_labels)
            outputs = discriminator(fake_imgs.detach(), class_labels)
            d_loss_fake = criterion(outputs, fake_labels)
            
            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            optimizer_D.step()

            # --- Train Generator ---
            optimizer_G.zero_grad()
            outputs = discriminator(fake_imgs, class_labels)
            g_loss = criterion(outputs, real_labels) # Trick D into thinking they are real
            g_loss.backward()
            optimizer_G.step()
            
        if (epoch+1) % 20 == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS}] D_Loss: {d_loss.item():.4f} G_Loss: {g_loss.item():.4f}")

    # 4. Save with specific name
    # Checks if we are running from root or inside experiments/
    if os.getcwd().endswith('experiments'):
        save_path = f"../cgan_generator_{digit}.pth"
    else:
        save_path = f"cgan_generator_{digit}.pth"
        
    torch.save(generator.state_dict(), save_path)
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--digit", type=int, default=0, help="Digit to train on (0-9)")
    args = parser.parse_args()
    
    train(args.digit)