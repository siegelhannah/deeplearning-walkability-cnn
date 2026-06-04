'''
walkability class prediction CNN model architecture
using Pytorch Lightning module to define training loop ahead of time, making train_model.py simpler
'''

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import Accuracy, ConfusionMatrix

class CNN(pl.LightningModule):

    def __init__(self, num_classes=3, learning_rate=1e-4): # 3 output classes
        depth=16 # depth size based on # kernels applied # 32 -> 16
        self.save_hyperparameters()
        super().__init__()

        # input images: 224x224
        # conv block 1
        self.conv1_1 = nn.Conv2d(3, depth, 3, stride=1, padding='same') # 3 RGB channels -> 32 channels (via 64 kernels)
        self.conv1_2 = nn.Conv2d(depth, depth, 3, stride=1, padding='same') # 32->32
        self.bn1 = nn.BatchNorm2d(depth) # batch normalization after each conv layer (twice)
        self.pool1   = nn.MaxPool2d(2, 2) # first pool: spatial size 224 -> 112

        # conv block 2
        self.conv2_1 = nn.Conv2d(depth, depth*2, 3, stride=1, padding='same') # 32->64
        self.conv2_2 = nn.Conv2d(depth*2, depth*2, 3, stride=1, padding=1) # 64->64
        self.bn2 = nn.BatchNorm2d(depth*2) # batch normalization after each conv layer (twice)
        self.pool2   = nn.MaxPool2d(2, 2) # spatial size 112 -> 56

        # # conv block 3
        # self.conv3_1 = nn.Conv2d(depth*2, depth*4, 3, padding='same') # 64->128
        # self.conv3_2 = nn.Conv2d(depth*4, depth*4, 3, padding='same') # 128->128
        # self.bn3 = nn.BatchNorm2d(depth*4) # batch normalization after each conv layer (twice)
        # self.pool3   = nn.MaxPool2d(2, 2) # spatial size 56->28
        
        # collapse spatial dims to fixed size before FC layers
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))  # collapses entire spatial dim

        # FC layers: 128 * 4 * 4 = 2048 -> 256 -> 3
        self.fc1 = nn.Linear(depth*4, 256) # just 128 features
        self.fc2 = nn.Linear(256, num_classes) # 256->10
        # DROPOUT to prevent overfitting (too high = inflated validation loss)
        self.dropout = nn.Dropout(0.01) # 0.3 -> 0.1


        # METRICS
        self.train_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_accuracy   = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_accuracy  = Accuracy(task="multiclass", num_classes=num_classes)
        self.confusion_matrix = ConfusionMatrix(task="multiclass", num_classes=num_classes)


    def forward(self, x):

        # conv block 1
        x = F.relu(self.bn1(self.conv1_1(x)))
        x = F.relu(self.bn1(self.conv1_2(x)))
        x = self.pool1(x)
        
        # conv block 2
        x = F.relu(self.bn2(self.conv2_1(x)))
        x = F.relu(self.bn2(self.conv2_2(x)))
        x = self.pool2(x)

        # conv block 3
        x = F.relu(self.bn3(self.conv3_1(x)))
        x = F.relu(self.bn3(self.conv3_2(x)))
        x = self.pool3(x)

        x = self.adaptive_pool(x) # 4x4 size
        x = x.view(x.size(0), -1) # flatten

        # fully connected layers
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)

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
        acc = self.train_accuracy(preds, y)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_acc", acc, on_step=True, on_epoch=True, prog_bar=True)

        return loss


    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)

        preds = torch.argmax(logits, dim=1)
        acc = self.val_accuracy(preds, y)

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

    
    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)

        preds = torch.argmax(logits, dim=1)
        self.confusion_matrix.update(preds, y)  # UPDATE CONFUSION MATRIX
        acc = self.test_accuracy(preds, y)

        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", acc, prog_bar=True)


    def on_test_epoch_end(self):
        # print confusion matrix after all test batches are done
        cm = self.confusion_matrix.compute()
        print("\nConfusion Matrix (rows=actual, cols=predicted)")
        print(cm)
        self.confusion_matrix.reset()


    def configure_optimizers(self):
        # optimizer for how model weights are updated during training
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
    
