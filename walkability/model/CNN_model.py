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

    def __init__(self, num_classes=3): # 3 output classes
        size=64
        super().__init__()
        self.conv1_1 = nn.Conv2d(3, size, 3, stride=1, padding=1) # 3 RGB channels -> 64 channels (via 64 kernels), kernel size=3x3
        self.conv1_2 = nn.Conv2d(size, size, 3, stride=1, padding=1)

        self.conv2_1 = nn.Conv2d(size, size*2, 3, stride=1, padding=1) # double in size: 128 channels -> 256
        self.conv2_2 = nn.Conv2d(size*2, size*2, 3, stride=1, padding=1)
        
        self.conv3 = nn.Conv2d(size*2, size*4, 3, stride=1, padding=1) # double again in size
        
        self.pool  = nn.MaxPool2d(2, 2) # 32->16->8
        # After two 2x2 pools on 28x28, spatial size is 7x7

        self.fc1   = nn.Linear(size*4 * 4 * 4 , size)
        self.fc2   = nn.Linear(size, size)
        self.fc3   = nn.Linear(size , num_classes)


    def forward(self, x):
        x = F.relu(self.conv1_1(x))
        x = F.relu(self.conv1_2(x))
        x = self.pool(x)
        
        x = F.relu(self.conv2_1(x))
        x = F.relu(self.conv2_2(x))
        x = self.pool(x)
        
        x = F.relu(self.conv3(x))
        x = self.pool(x)

        x = x.view(x.size(0), -1) # Flatten

        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=self.fc3(x)

        return x
    


    def training_step(self, batch, batch_idx): 
        '''
        What happens in one step of the training loop (calculating loss, etc.)
        '''
        x, y = batch # batch includes x: images and y: labels
        # LOSS:
        logits = self(x) # make prediction
        loss = F.cross_entropy(logits, y) # calculate loss ( same as nn.CrossEntropyLoss() )
        # ACCURACY:
        preds = torch.argmax(logits, dim=1)
        acc = self.train_accuracy(preds, y) # TODO: define train_accuracy metric in __init__()

        # TODO: add logs/prints for loss & accuracy?

        return loss



    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)

        preds = torch.argmax(logits, dim=1)
        acc = self.val_accuracy(preds, y) # TODO: define val_accuracy metric in __init__()

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)



    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)

        preds = torch.argmax(logits, dim=1)
        acc = self.test_accuracy(preds, y) # TODO: define test_accuracy metric in __init__()

        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", acc, prog_bar=True)



    def configure_optimizers(self):
        ...
        return
    
