import torch
from torch.utils.data import Dataset
from torchvision import transforms

class AugmentedDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        
        # Phase 1: Geometric Augmentations
        self.transform = transforms.Compose([
            transforms.ToPILImage(),       
            transforms.RandomRotation(15), 
            transforms.RandomAffine(0, translate=(0.1, 0.1)), 
            transforms.ToTensor()          
        ])

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # Reshape flat vector to (1, 28, 28) -> Augment -> Flatten back
        img_flat = self.X[idx]
        label = self.y[idx]
        img_2d = img_flat.view(1, 28, 28)
        img_aug = self.transform(img_2d)
        return img_aug.view(-1), label