# models/cgan.py
import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, z_dim=100, num_classes=10, img_dim=784):
        super().__init__()
        self.label_emb = nn.Embedding(num_classes, num_classes)
        
        self.model = nn.Sequential(
            nn.Linear(z_dim + num_classes, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, img_dim),
            nn.Tanh()  # Outputs image pixels between -1 and 1
        )

    def forward(self, noise, labels):
        # Concatenate Noise + Label so the GAN knows WHAT to draw
        gen_input = torch.cat((noise, self.label_emb(labels)), -1)
        img = self.model(gen_input)
        return img

class Discriminator(nn.Module):
    def __init__(self, num_classes=10, img_dim=784):
        super().__init__()
        self.label_emb = nn.Embedding(num_classes, num_classes)
        
        self.model = nn.Sequential(
            nn.Linear(img_dim + num_classes, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img, labels):
        d_input = torch.cat((img, self.label_emb(labels)), -1)
        validity = self.model(d_input)
        return validity