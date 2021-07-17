# PyTorch Lightning Cifar10
The code is a simple example that shows the basic use of pytorch-lightning in CIFAR10 dataset. It includes the training, testing, validating, saving checkpoints, and logging of the model. Some code is from https://pytorch-lightning.readthedocs.io/. The copyright of the code belongs to the original authors of the code.

## Quickstart
Firstly, you should install NCCL on your machine.

Enter the directory that you want to save the code.
```bash
https://github.com/zhoufengfan/pytorch-lightning-cifar10.git
cd pytorch-lightning-cifar10
conda create -n pytorch-lightning-cifar10 python=3.8.5
conda activate pytorch-lightning-cifar10
pip intall -r requirements.txt
```
Then download the CIFAR10 dataset from https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz. Then use the following code to decompress it:
```bash
tar -xzf cifar-10-python.tar.gz
```

Edit the CIFAR10.cifar10_root_dir in `dataset.py` to the path of your dataset.

Run the following command to start your experiment:
``` bash
python train.py
```

## Reuslt
Test accuracy: 0.83170

If you have any questions, feel free to post an issue.

## Note
By default, the model will be trained in fp16. If you don't want to train the model in fp16, you can delete `precision=16` in `train.py`.