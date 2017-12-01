#!/usr/bin/env bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

echo "Downloading and unpacking CIFAR-10"
mkdir -p $DIR/../workdir
python $DIR/unpack_cifar10.py $DIR/../workdir $DIR/../images/cifar/cifar10/by-image/

echo "Linking training set"
(
    cd $DIR/../images/cifar/cifar10/by-image/
    bash $DIR/link_cifar10_train.sh
)

echo "Linking validation set"
(
    cd $DIR/../images/cifar/cifar10/by-image/
    bash $DIR/link_cifar10_val.sh
)
