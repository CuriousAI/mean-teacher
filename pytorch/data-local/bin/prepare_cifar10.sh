#!/usr/bin/env bash
#
# Copyright (c) 2018, Curious AI Ltd. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

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
