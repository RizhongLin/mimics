import os
from datetime import datetime
import torch
import wandb
from pytorch_lightning.loggers import WandbLogger
from data import *
from model import *

if __name__ == '__main__':
    early_stopping = pl.callbacks.early_stopping.EarlyStopping(
        monitor='val_loss',
    )

    trainer = pl.Trainer(
        accelerator='gpu',
        precision=16,
        callbacks=[early_stopping],
        max_epochs=-1,
        # logger=logger,
        log_every_n_steps=10,
    )

    data = BraTS2019DataModule()
    model = EvidentialUNet(
        learning_rate=0.002, net_config="default", optimizer_class=torch.optim.Adam)

    start = datetime.now()
    print('Training started at', start)
    trainer.fit(model=model, datamodule=data)
    print('Training duration:', datetime.now() - start)
