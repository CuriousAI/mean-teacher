#!/usr/bin/env bash

mkdir -p data/images/svhn
(
    cd data/images/svhn
    curl -O 'http://ufldl.stanford.edu/housenumbers/{train,test,extra}_32x32.mat'
)
