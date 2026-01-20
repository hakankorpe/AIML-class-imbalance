import torch
import matplotlib.pyplot as plt
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from models.cgan import Generator

def visualize_samples():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    z_dim = 100
    
    # Load Generator
    generator = Generator(z_dim, num_classes=2).to(device)
    try:
        generator.load_state_dict(torch.load("cgan_generator.pth", map_location=device))
    except FileNotFoundError:
        print("Model not found.")
        return

    generator.eval()
    
    # Generate 16 fake "Zeros"
    num_samples = 16
    noise = torch.randn(num_samples, z_dim).to(device)
    labels = torch.ones(num_samples).long().to(device) # Label 1 = "Zero"
    
    with torch.no_grad():
        fake_imgs = generator(noise, labels).cpu().view(-1, 28, 28)

    # Plot
    fig, axes = plt.subplots(2, 8, figsize=(12, 3))
    for i, ax in enumerate(axes.flatten()):
        ax.imshow(fake_imgs[i], cmap='gray')
        ax.axis('off')
    
    plt.suptitle("Generated Synthetic 'Zeros' (GAN)", fontsize=16)
    plt.savefig("figures/gan_samples.png")
    print("Saved GAN samples to figures/gan_samples.png")

if __name__ == "__main__":
    visualize_samples()