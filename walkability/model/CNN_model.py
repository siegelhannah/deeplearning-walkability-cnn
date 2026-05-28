'''
walkability class prediction CNN model architecture
using Pytorch Lightning module to define training loop ahead of time, making train_model.py simpler
'''

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import Accuracy
from torchvision import models


class ImageClassifier(pl.LightningModule):
    def __init__(self, num_classes, learning_rate=1e-3):
        super().__init__()

        # model architecture: ...



    def forward(self, x):
        ...

    def training_step(self, batch, batch_idx):
        ...
        return loss


    def validation_step(self, batch, batch_idx):
        ...

    def test_step(self, batch, batch_idx):
        ...

    def configure_optimizers(self):
        ...
        return
    
