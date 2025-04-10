import pytorch_lightning as pl  # noqa

from skeleplex.measurements.lumen_classifier import (
    H5DataModule,
    ResNet3ClassClassifier,
)

if __name__ == "__main__":
    dataset_root = "dataset/"
    # needs to contain folders train and val
    # each of them needs to contain h5 files with an image and class_id key

    model = ResNet3ClassClassifier(num_classes=3)
    data_module = H5DataModule(data_dir=dataset_root, batch_size=16)

    # logging
    logger = pl.loggers.TensorBoardLogger("logs/", name="resnet3class")

    # early stopping
    early_stop_callback = pl.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=10,
        verbose=True,
        mode="min",
    )
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor="val_loss",
        dirpath="checkpoints/",
        filename="resnet3class-{epoch:02d}-{val_loss:.2f}",
        save_top_k=1,
        mode="min",
    )
    trainer = pl.Trainer(
        max_epochs=200,
        logger=logger,
        accelerator="gpu",
        callbacks=[early_stop_callback, checkpoint_callback],
    )

    # Train the model
    trainer.fit(model, data_module)
