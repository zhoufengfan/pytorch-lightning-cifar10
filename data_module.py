# The code of this file is from `https://pytorch-lightning.readthedocs.io/en/latest/starter/new-project.html`. The copyright of this file belongs to the original authors of this file.
from torchvision import transforms
from dataset import CIFAR10
# from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl

from hyper_var import BATCH_SIZE, NUM_WORKERS


class CIFAR10DataModule(pl.LightningDataModule):

    def __init__(self):
        super().__init__()
        # self.data_dir = data_dir
        self.transform = transforms.Compose([
            transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        self.dims = (3, 32, 32)
        self.num_classes = 10

    def prepare_data(self):
        # # download
        # CIFAR10(root=PATH_DATASETS, train=True, download=True)
        # CIFAR10(root=PATH_DATASETS, train=False, download=True)
        pass

    def setup(self, stage=None):

        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            cifar_full = CIFAR10(train=True, transform=self.transform)
            # cifar_full = CIFAR10(root=PATH_DATASETS, train=True, transform=self.transform)
            self.cifar_train, self.cifar_val = random_split(cifar_full, [45000, 5000])

        # Assign test dataset for use in dataloader(s)
        if stage == 'test' or stage is None:
            self.cifar_test = CIFAR10(train=False, transform=self.transform)
            # self.cifar_test = CIFAR10(root=PATH_DATASETS, train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.cifar_train, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

    def val_dataloader(self):
        return DataLoader(self.cifar_val, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

    def test_dataloader(self):
        return DataLoader(self.cifar_test, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)


cifar10_dm = CIFAR10DataModule()
