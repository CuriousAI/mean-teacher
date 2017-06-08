#!/usr/bin/env bash

echo "Downloading SVHN"
mkdir -p data/images/svhn
(
    cd data/images/svhn
    curl -O 'http://ufldl.stanford.edu/housenumbers/{train,test,extra}_32x32.mat'
)

echo
echo "Downloading CIFAR-10"
mkdir -p data/images/cifar/cifar10
(
    cd data/images/cifar/cifar10
    curl -O 'https://www.cs.toronto.edu/~kriz/cifar-10-matlab.tar.gz'
    tar xvzf cifar-10-matlab.tar.gz
    mv cifar-10-batches-mat/* .
    rmdir cifar-10-batches-mat
)

echo
echo "Preprocessing CIFAR-10"
python datasets/preprocess_cifar10.py

echo
echo "All done!"
