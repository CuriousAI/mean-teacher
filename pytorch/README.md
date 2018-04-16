# Mean Teacher using PyTorch

This is the PyTorch source code for the Mean Teacher paper. The code runs on Python 3. Install the dependencies and prepare the datasets with the following commands:

```
pip install numpy scipy pandas pytorch tqdm matplotlib
pip install git+ssh://git@github.com/pytorch/vision@c31c3d7e0e68e871d2128c8b731698ed3b11b119
```

The code expects to find the data in specific directories inside the data-local directory. You can prepare the CIFAR-10 with this command:

```
./data-local/bin/prepare_cifar10.sh
```

You can prepare the ImageNet using [these instructions](https://github.com/facebook/fb.resnet.torch/blob/master/INSTALL.md#download-the-imagenet-dataset) (Section "Download the ImageNet dataset"). The mean teacher code expects to find the ImageNet data at `data-local/images/ilsvrc2012/`.

To train on CIFAR-10, run e.g.:

```
python main.py \
    --dataset cifar10 \
    --labels data-local/labels/cifar10/1000_balanced_labels/00.txt \
    --arch cifar_shakeshake26 \
    --consistency 100.0 \
    --consistency-rampup 5 \
    --labeled-batch-size 62 \
    --epochs 180 \
    --lr-rampdown-epochs 210
```

Use `python main.py --help` to see other command line arguments.

To reproduce the CIFAR-10 ResNet results of the paper run `python -m experiments.cifar10_test` using 4 GPUs.

To reproduce the ImageNet results of the paper run `python -m experiments.imagenet_valid` using 10 GPUs.
