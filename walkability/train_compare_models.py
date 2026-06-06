"""
Code to train and compare models:
- From-scratch CNN
- ResNet18 (fine-tuned)
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from walkability.dataset.dataloader import get_data_loaders
from walkability.train_model import train_model

# dataset
BASE_PATH = "/projects/dsci410_510/data/walkability_dataset/"

# load data
train_loader, val_loader, test_loader = get_data_loaders(base_path=BASE_PATH, name="all", batch_size=32)


# train both models
architectures = ["scratch", "resnet"]
results = {}

for arch in architectures:
    model, trainer = train_model(train_loader, val_loader, test_loader, architecture=arch, max_epochs=100)

    # get test metrics logged by Lightning
    test_results = trainer.callback_metrics
    results[arch] = {
        "test_acc":  float(test_results.get("test_acc",  0)),
        "test_loss": float(test_results.get("test_loss", 0)),
    }

# print summary table
print("RESULTS SUMMARY")
print(f"{'Model':<20} {'Test Acc':>10} {'Test Loss':>10}")
for arch, metrics in results.items():
    print(f"{arch:<20} {metrics['test_acc']:>10.4f} {metrics['test_loss']:>10.4f}")

# Plot training curves for each model 
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle("Training Comparison: Scratch vs Pretrained", fontsize=16, fontweight="bold")


for col, arch in enumerate(architectures):
    # Lightning saves logs to lightning_logs/version_0, version_1, version_2 in order
    log_path = f"lightning_logs/version_{col}/metrics.csv"
    if not os.path.exists(log_path):
        print(f"Warning: no log found at {log_path}")
        continue


    metrics = pd.read_csv(log_path)

    # loss
    train_loss = metrics.dropna(subset=["train_loss"])
    val_loss   = metrics.dropna(subset=["val_loss"])
    axes[0][col].plot(train_loss["epoch"], train_loss["train_loss"], label="train")
    axes[0][col].plot(val_loss["epoch"],   val_loss["val_loss"],           label="val")
    axes[0][col].set_title(f"{arch} — Loss")
    axes[0][col].set_xlabel("Epoch")
    axes[0][col].set_ylabel("Loss")
    axes[0][col].legend()

    # accuracy
    train_acc = metrics.dropna(subset=["train_acc"])
    val_acc   = metrics.dropna(subset=["val_acc"])
    axes[1][col].plot(train_acc["epoch"], train_acc["train_acc"], label="train")
    axes[1][col].plot(val_acc["epoch"],   val_acc["val_acc"],           label="val")
    axes[1][col].set_title(f"{arch} — Accuracy")
    axes[1][col].set_xlabel("Epoch")
    axes[1][col].set_ylabel("Accuracy")
    axes[1][col].legend()



plt.tight_layout()
plt.savefig("logs/training_curves.png", dpi=150, bbox_inches="tight")
print("\nTraining curves saved to logs/training_curves.png")