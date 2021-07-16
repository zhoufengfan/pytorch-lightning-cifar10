from torch.utils.data import Dataset
import torch
import pickle
import os
import numpy as np
from PIL import Image

class CIFAR10(Dataset):
    def __init__(self, train, transform):
        super().__init__()
        self.cifar10_root_dir = r"/media/data1/datasets/cifar-10-batches-py"
        self.is_train = train
        self.transform = transform
        self.imgs, self.labels = self.get_data_from_pickled_files(self.is_train)

    def unpickle(self, file):
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='latin1')
        return dict

    def get_data_from_pickled_files(self, is_train):
        imgs = []
        labels = []
        if is_train:
            for i in range(1, 6):
                dict_of_single_pickled_file = self.unpickle(
                    os.path.join(self.cifar10_root_dir, "data_batch_{}".format(str(i))))
                imgs.append(dict_of_single_pickled_file['data'])
                labels.extend(dict_of_single_pickled_file['labels'])
        else:
            dict_of_single_pickled_file = self.unpickle(
                os.path.join(self.cifar10_root_dir, "test_batch"))
            imgs.append(dict_of_single_pickled_file['data'])
            labels.extend(dict_of_single_pickled_file['labels'])

        imgs = np.vstack(imgs).reshape((-1, 3, 32, 32))
        imgs = imgs.transpose((0, 2, 3, 1))
        labels = np.array(labels)
        return imgs, labels

    def __getitem__(self, item):
        img, label = self.imgs[item], self.labels[item]
        img = Image.fromarray(img)
        img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.long)

    def __len__(self):
        return len(self.labels)
