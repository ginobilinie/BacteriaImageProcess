#!/usr/bin/env sh
# Compute the mean image from the imagenet training lmdb
# N.B. this is available in data/ilsvrc12

EXAMPLE=.
DATA=.
TOOLS=../../build/tools

$TOOLS/compute_image_mean $EXAMPLE/bacteria_train_lmdb \
  $DATA/bacteria_mean.binaryproto

echo "Done."
