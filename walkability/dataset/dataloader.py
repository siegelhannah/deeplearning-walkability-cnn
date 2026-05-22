import os
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from pathlib import Path

# labels: string -> integer for PyTorch
LABEL_MAP = {"low": 0, "medium": 1, "high": 2}

# ImageNet normalization
DEFAULT_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Training transform: image augmentations (flips, colors)
TRAIN_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# DATASET CLASS:
class WalkabilityDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        """
        Args:
            dataframe: DataFrame with columns for image path and label
            transform: torchvision transforms to apply to each image
        """
        self.data = dataframe.reset_index(drop=True)
        self.transform = transform if transform is not None else DEFAULT_TRANSFORM

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.data.iloc[idx]["path"] # image path column for that sample
        image = Image.open(img_path).convert("RGB")
        label = LABEL_MAP[self.data.iloc[idx]["label"]]  # string -> int

        if self.transform:
            image = self.transform(image)

        return image, label


# PYTORCH DATALOADERS:
def get_data_loaders(base_path, name="all", batch_size=32):
    """
    Create data loaders for train, develop/validation, and test sets.
 
    Args:
        base_path: Path to the walkability dataset directory.
                   On Talapas: /projects/dsci410_510/data/walkability_dataset
                   Must contain train.csv, develop.csv, test.csv
        name: One of 'train', 'develop', 'test', 'all'
        batch_size: Batch size for the dataloaders
 
    Returns:
        A single DataLoader if name is 'train', 'develop', or 'test'.
        A tuple (train_loader, val_loader, test_loader) if name is 'all'.
 
    Example usage:
        train_loader, val_loader, test_loader = get_data_loaders(
            base_path="/projects/dsci410_510/data/walkability_dataset",
            name="all"
        )
    """
    if name in ["train", "all"]:
        train_df = pd.read_csv(os.path.join(base_path, "train.csv"))
        train_loader = DataLoader(
            WalkabilityDataset(train_df, transform=TRAIN_TRANSFORM),
            batch_size=batch_size, shuffle=True, num_workers=1
        )
    if name == "train":
        return train_loader
 
    if name in ["develop", "validation", "all"]:
        val_df = pd.read_csv(os.path.join(base_path, "develop.csv"))
        val_loader = DataLoader(
            WalkabilityDataset(val_df, transform=DEFAULT_TRANSFORM),
            batch_size=batch_size, shuffle=False, num_workers=4
        )
    if name in ["develop", "validation"]:
        return val_loader
 
    if name in ["test", "all"]:
        test_df = pd.read_csv(os.path.join(base_path, "test.csv"))
        test_loader = DataLoader(
            WalkabilityDataset(test_df, transform=DEFAULT_TRANSFORM),
            batch_size=batch_size, shuffle=False, num_workers=4
        )
    if name == "test":
        return test_loader
 
    return train_loader, val_loader, test_loader


# RUN EVERYTHING: 
if __name__ == "__main__":
    import sys
    # path where dataset is stored: should be hosted on /projects/dsci410_510/data/walkability_dataset
    base_path = sys.argv[1] if len(sys.argv) > 1 else "/projects/dsci410_510/data/walkability_dataset"
    
    # load dataloaders:
    train_loader, val_loader, test_loader = get_data_loaders(base_path, name="all")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches:   {len(val_loader)}")
    print(f"Test batches:  {len(test_loader)}")
    for images, labels in train_loader:
        print(f"Batch shape: {images.shape}, Labels: {labels[:8]}")
        break
