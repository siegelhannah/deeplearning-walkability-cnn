# existing + fine-tuned model architecture
# resnet18: Smallest ResNet, fast to fine-tune, well understood
# to compare with built-from-scratch one

from torchvision import models as tv_models
from torchgeo import models as tg_models # for satellite-imagery trained resnet weights
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import Accuracy, ConfusionMatrix


class Pretrained_Resnet(pl.LightningModule):
    def __init__(self, num_classes=3, learning_rate=1e-5): # smaller lr than scratch model
        super().__init__()
        self.save_hyperparameters()

        self.model = tv_models.resnet18(weights=tv_models.ResNet18_Weights.DEFAULT) # weights

        # FREEZE ARCHITECTURE: Turn off gradients for all layers
        for param in self.model.parameters():
            param.requires_grad = False

        # new fc layer (head) WITH DROPOUT
        self.model.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.model.fc.in_features, num_classes)
            )

        # unfreeze only the fc classifier head
        for param in self.model.fc.parameters():
            param.requires_grad = True

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
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_acc", acc, on_step=False, on_epoch=True, prog_bar=True)
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
        # FILTER PARAMETERS: Only give active gradients to the optimizer
        trainable_params = filter(lambda p: p.requires_grad, self.parameters())
        return torch.optim.Adam(trainable_params, lr=self.hparams.learning_rate)
        