### walkability class prediction training loop

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from walkability.models.built_cnn import CNN
from walkability.models.finetuned_resnet import Pretrained_Resnet


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
        enable_progress_bar=True,
        enable_model_summary=True,
        log_every_n_steps=10,
        # callbacks=[early_stopping], # no callbacks for debugging
    )

    trainer.fit(model, train_loader, val_loader)
    trainer.test(model, test_loader)

    return model, trainer