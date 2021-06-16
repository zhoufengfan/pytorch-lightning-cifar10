# The code is from `https://pytorch-lightning.readthedocs.io/en/latest/notebooks/starters/cifar10-baseline.html`. The copyright of this file belongs to the original authors of this file.
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger

from data_module import cifar10_dm
from hyper_var import AVAIL_GPUS
from backbone import LitResnet

if __name__ == '__main__':
    model = LitResnet(lr=2e-3)
    model.datamodule = cifar10_dm

    trainer = Trainer(
        progress_bar_refresh_rate=10,
        max_epochs=60,
        gpus=AVAIL_GPUS,
        logger=TensorBoardLogger('lightning_logs/', name='resnet'),
        accelerator='ddp'
    )

    trainer.fit(model, cifar10_dm)
    trainer.test(model, datamodule=cifar10_dm)
