#!/usr/bin/env bash

# Given a directory with ImageFolder structure and number of labels per class,
# create a list of image filenames and their labels..
#
# Usage: scripts/create_balanced_semisupervised_labels.sh DATADIR LABELS_PER_CLASS

DIR=$1
LABELS_PER_CLASS=$2

(
    cd $DIR/train                   # Go to the train directory
    find * -maxdepth 0 -type d |    # Find the class directories
    xargs -n 1 -i{} sh -c  "        # Loop through the classes names, running the following for each:
        ls {} |                     # List all files in the class directory
        shuf -n $LABELS_PER_CLASS | # Shuffle and pick first N
        sed -e 's!\$!\ {}!'         # Add the class label to the output
    "
)
