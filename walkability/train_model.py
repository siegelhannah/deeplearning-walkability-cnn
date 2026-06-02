### walkability class prediction training loop

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from walkability.models.built_cnn import CNN
from walkability.models.finetuned_resnet import Pretrained_Resnet

from pytorch_lightning.callbacks import Callback

# for simplified epoch progress logging:
class EpochSummary(Callback):
    def on_validation_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        epoch = trainer.current_epoch
        train_loss = metrics.get("train_loss_epoch", float("nan"))
        val_loss   = metrics.get("val_loss",         float("nan"))
        train_acc  = metrics.get("train_acc_epoch",  float("nan"))
        val_acc    = metrics.get("val_acc",          float("nan"))
        print(f"Epoch {epoch:3d} | train_loss: {train_loss:.4f} | val_loss: {val_loss:.4f} | train_acc: {train_acc:.4f} | val_acc: {val_acc:.4f}")


def train_model(train_loader, val_loader, test_loader, 
                num_classes=3, max_epochs=50, architecture="scratch"):
    """
    Trains model architecture specified (scratch, resnet) 
    """

    # select model
    if architecture == "scratch":
        model = CNN(num_classes=num_classes)
    elif architecture == "resnet":
        model = Pretrained_Resnet(num_classes=num_classes) # 224x224 imagery data is already ready for resnet18
    else:
        raise ValueError("architecture must be one of: scratch, resnet,")


    early_stopping = EarlyStopping(monitor="val_loss", patience=10, mode="min") # "min" mode for loss

    # pytorch lightning train loop
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="auto", # automatically uses GPU if available
        devices=1,
        enable_progress_bar=False, # to write less to .out
        enable_model_summary=True,
        log_every_n_steps=50, # less logs
        callbacks=[EpochSummary()] #, early_stopping] # less epochs # don't include early_stopping yet
    )

    trainer.fit(model, train_loader, val_loader)
    trainer.test(model, test_loader)

    return model, trainer