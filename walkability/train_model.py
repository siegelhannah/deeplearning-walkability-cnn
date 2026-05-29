### walkability class prediction training loop

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from walkability.model.CNN_model import ImageClassifier


def train_model(train_loader, val_loader, test_loader, num_classes=3, max_epochs=30): # TODO: max_epochs
    model = ImageClassifier(num_classes=num_classes)

    early_stopping = EarlyStopping(monitor="val_loss", patience=5, mode="min")

    # pytorch lightning train loop
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="auto",
        devices=1,
        enable_progress_bar=True,
        enable_model_summary=True,
        log_every_n_steps=10,
        callbacks=[early_stopping],
    )

    trainer.fit(model, train_loader, val_loader)
    trainer.test(model, test_loader)

    return model, trainer