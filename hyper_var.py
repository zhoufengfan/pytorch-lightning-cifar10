# The code of this file is from `https://pytorch-lightning.readthedocs.io`. The copyright of this file belongs to the original authors of this file.
import os
import torch
import torch.nn as nn
import torchvision
from pl_bolts.transforms.dataset_normalizations import cifar10_normalization
from pytorch_lightning import seed_everything

seed_everything(7)

PATH_DATASETS = os.environ.get('PATH_DATASETS', '.')
AVAIL_GPUS = "2"
# AVAIL_GPUS = min(1, torch.cuda.device_count())
# AVAIL_GPUS = "1,2"
BATCH_SIZE = 256 if AVAIL_GPUS else 64
NUM_WORKERS = int(os.cpu_count() / 2)

train_transforms = torchvision.transforms.Compose([
    torchvision.transforms.RandomCrop(32, padding=4),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    cifar10_normalization(),
])

test_transforms = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    cifar10_normalization(),
])


def create_model():
    model = torchvision.models.resnet18(pretrained=False, num_classes=10)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    model.maxpool = nn.Identity()
    return model
