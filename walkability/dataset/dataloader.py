import os
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from pathlib import Path
from sklearn.model_selection import train_test_split

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



def get_data_loaders(base_path="./", name="example", batch_size=32):
    """
    Create data loaders for train, develop, and test sets.
 
    Args:
        base_path (str): Path to directory containing train.csv, develop.csv, test.csv, and an example_dataset/ subfolder
        name (str): One of 'example', 'train', 'develop'/'validation', 'test', 'all'
        batch_size (int): Batch size for the dataloaders
 
    Returns:
        A single DataLoader if name is 'example', 'train', 'develop', or 'test'.
        A tuple (train_loader, val_loader, test_loader) if name is 'all'.
 
    Example usage:
        # load all splits (typical for training)
        train_loader, val_loader, test_loader = get_data_loaders(
            base_path="/path/to/walkability_dataset", name="all"
        )
 
        # load a single split
        train_loader = get_data_loaders(base_path="/path/to/walkability_dataset", name="train")
 
        # load example data bundled with the repo (no dataset download needed)
        example_loader = get_data_loaders(name="example")
    """
 
    # example mode: loads a few sample images bundled in the repo for quick testing
    if name == "example":
        example_path = Path(__file__).parent / ".." / ".." / "tests" / "example_dataset"
        df = pd.read_csv(example_path / "example.csv")
        dataset = WalkabilityDataset(df, transform=DEFAULT_TRANSFORM)
        return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
 
    if name in ["train", "all"]:
        train_df = pd.read_csv(os.path.join(base_path, "train.csv"))
        train_loader = DataLoader(
            WalkabilityDataset(train_df, transform=TRAIN_TRANSFORM),
            batch_size=batch_size, shuffle=True, num_workers=4
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


 
if __name__ == "__main__":
    # quick test using example data bundled with the repo
    example_loader = get_data_loaders(name="example")
    for images, labels in example_loader:
        print(f"Example batch — images: {images.shape}, labels: {labels}")
        break
 
