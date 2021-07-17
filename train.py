# The code is from `https://pytorch-lightning.readthedocs.io/en/latest/notebooks/starters/cifar10-baseline.html`. The copyright of this file belongs to the original authors of this file.
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.plugins import DDPPlugin

from data_module import cifar10_dm
from hyper_var import AVAIL_GPUS
from backbone import LitResnet

if __name__ == '__main__':
    model = LitResnet(lr=2e-3)
    model.datamodule = cifar10_dm
    checkpoint_callback = ModelCheckpoint(dirpath='lightning_logs/',monitor='val_acc')
    trainer = Trainer(
        progress_bar_refresh_rate=10,
        max_epochs=1,
        gpus=AVAIL_GPUS,
        callbacks=[checkpoint_callback],
        accelerator='ddp',
        plugins=DDPPlugin(find_unused_parameters=False),
        precision=16
    )

    trainer.fit(model, cifar10_dm)
    trainer.test(model, datamodule=cifar10_dm)
