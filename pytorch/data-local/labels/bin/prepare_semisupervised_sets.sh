#!/usr/bin/env bash

# Prepare semisupervised datasets needed in the experiments

SCRIPT=scripts/create_balanced_semisupervised_labels.sh

create ()
{
    for LABELS_PER_CLASS in ${LABEL_VARIATIONS[@]}
    do
        LABELS_IN_TOTAL=$(( $LABELS_PER_CLASS * $NUM_CLASSES ))
        echo "Creating sets for $DATANAME with $LABELS_IN_TOTAL labels."
        for IDX in {00..19}
        do
            LABEL_DIR=labels/${DATANAME}/${LABELS_IN_TOTAL}_balanced_labels
            mkdir -p $LABEL_DIR
            $SCRIPT $DATADIR $LABELS_PER_CLASS > $LABEL_DIR/${IDX}.txt
        done
    done
}

DATADIR=data/images/cifar/cifar10/by-image/
DATANAME=cifar10
NUM_CLASSES=10
LABEL_VARIATIONS=(100 400)
create

DATADIR=data/images/ilsvrc2012/
DATANAME=ilsvrc2012
NUM_CLASSES=1000
LABEL_VARIATIONS=(128)
create
