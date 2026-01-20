import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import sys
import os

# Fix imports to look at parent directory
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from data.load_mnist_binary import load_mnist_binary_digit
from models.cgan import Generator, Discriminator

def run_gan_training():
    # --- Settings ---
    z_dim = 100
    lr = 0.0002
    batch_size = 64
    epochs = 200 # Higher epochs needed because we have very little data!
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_classes = 2 # 0 vs 1 (Binary task)

    print(f"Training GAN on {device} using imbalanced data...")

    # 1. Load Data (Imbalanced 2% split)
    # y_train is 0 (Majority) or 1 (Minority/Zero)
    X_train, _, y_train, _ = load_mnist_binary_digit(minority_digit=0, minority_ratio=0.02)
    
    # 2. Preprocess: [0, 1] -> [-1, 1] for Tanh activation
    X_train = (X_train * 2) - 1.0
    
    # 3. Create Loader
    tensor_x = torch.tensor(X_train).float()
    tensor_y = torch.tensor(y_train).long()
    loader = DataLoader(TensorDataset(tensor_x, tensor_y), batch_size=batch_size, shuffle=True)

    # 4. Initialize Models
    generator = Generator(z_dim, num_classes=num_classes).to(device)
    discriminator = Discriminator(num_classes=num_classes).to(device)
    
    opt_g = optim.Adam(generator.parameters(), lr=lr)
    opt_d = optim.Adam(discriminator.parameters(), lr=lr)
    criterion = nn.BCELoss()

    # 5. Training Loop
    for epoch in range(epochs):
        for batch_idx, (real_imgs, labels) in enumerate(loader):
            real_imgs, labels = real_imgs.to(device), labels.to(device)
            curr_batch_size = real_imgs.shape[0]

            # --- Train Discriminator ---
            noise = torch.randn(curr_batch_size, z_dim).to(device)
            fake_imgs = generator(noise, labels)
            
            # Real Loss
            real_preds = discriminator(real_imgs, labels)
            real_loss = criterion(real_preds, torch.ones_like(real_preds))
            
            # Fake Loss
            fake_preds = discriminator(fake_imgs.detach(), labels)
            fake_loss = criterion(fake_preds, torch.zeros_like(fake_preds))
            
            d_loss = (real_loss + fake_loss) / 2
            opt_d.zero_grad()
            d_loss.backward()
            opt_d.step()

            # --- Train Generator ---
            output = discriminator(fake_imgs, labels)
            g_loss = criterion(output, torch.ones_like(output))
            
            opt_g.zero_grad()
            g_loss.backward()
            opt_g.step()

        if epoch % 20 == 0:
            print(f"Epoch {epoch}/{epochs} | D Loss: {d_loss.item():.4f} | G Loss: {g_loss.item():.4f}")

    # 6. Save Model
    torch.save(generator.state_dict(), "cgan_generator.pth")
    print("GAN Saved to cgan_generator.pth")

if __name__ == "__main__":
    run_gan_training()