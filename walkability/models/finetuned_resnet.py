# existing + fine-tuned model architecture
# resnet18: Smallest ResNet, fast to fine-tune, well understood
# to compare with built-from-scratch one

from torchvision import models
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import Accuracy, ConfusionMatrix


class Pretrained_Resnet(pl.LightningModule):
    def __init__(self, num_classes=3, learning_rate=1e-5):
        super().__init__()
        self.save_hyperparameters()

        self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT) # weights
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

        # same metrics as from-scratch model
        self.train_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_accuracy   = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_accuracy  = Accuracy(task="multiclass", num_classes=num_classes)
        self.confusion_matrix = ConfusionMatrix(task="multiclass", num_classes=num_classes)


    def forward(self, x):
        return self.model(x)
    

    # training / validation / test step functions are the same as from-scratch cnn
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
        acc = self.test_accuracy(preds, y)

        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", acc, prog_bar=True)


    def on_test_epoch_end(self):
        # print confusion matrix after all test batches are done
        cm = self.confusion_matrix.compute()
        print("\nConfusion Matrix (rows=actual, cols=predicted)")
        labels = ["low", "med", "high"]
        for i, row in enumerate(cm):
            print(f"{labels[i]}  {row.tolist()}")
        self.confusion_matrix.reset()

    
    def configure_optimizers(self):
        # lower lr for fine-tuning than training from scratch
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        