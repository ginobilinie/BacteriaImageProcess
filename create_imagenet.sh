#!/usr/bin/env sh
# Create the imagenet lmdb inputs
# N.B. set the path to the imagenet train + val data dirs

EXAMPLE=.
DATA=/home/dongnie/Desktop/Caffes/caffe/data/bacteria/split
TOOLS=/home/dongnie/Desktop/Caffes/caffe/build/tools/

DATA_ROOT=/home/dongnie/Desktop/Caffes/data/bacteria/patchImages/
#DATA_ROOT is just a prefix, if you have specified the path in your file, you need not use it
#if there is no prepath in your train.txt or val.txt, you should use it
#TRAIN_DATA_ROOT=/path/to/imagenet/train/
#VAL_DATA_ROOT=/path/to/imagenet/val/

# Set RESIZE=true to resize the images to 256x256. Leave as false if images have
# already been resized using another tool.
RESIZE=false
if $RESIZE; then
  RESIZE_HEIGHT=256
  RESIZE_WIDTH=256
else
  RESIZE_HEIGHT=0
  RESIZE_WIDTH=0
fi

if [ ! -d "$DATA_ROOT" ]; then
  echo "Error: TRAIN_DATA_ROOT is not a path to a directory: $TRAIN_DATA_ROOT"
  echo "Set the TRAIN_DATA_ROOT variable in create_imagenet.sh to the path" \
       "where the ImageNet training data is stored."
  exit 1
fi

#if [ ! -d "$VAL_DATA_ROOT" ]; then
#  echo "Error: VAL_DATA_ROOT is not a path to a directory: $VAL_DATA_ROOT"
#  echo "Set the VAL_DATA_ROOT variable in create_imagenet.sh to the path" \
#       "where the ImageNet validation data is stored."
#  exit 1
#fi

#echo "Creating train lmdb..."

#GLOG_logtostderr=1 $TOOLS/convert_imageset \
#    --resize_height=$RESIZE_HEIGHT \
#    --resize_width=$RESIZE_WIDTH \
#    --shuffle \
#    $DATA_ROOT \
#    $DATA/newtrain.txt \
#    $EXAMPLE/bacteria_train_lmdb

echo "Creating val lmdb..."

GLOG_logtostderr=1 $TOOLS/convert_imageset \
    --resize_height=$RESIZE_HEIGHT \
    --resize_width=$RESIZE_WIDTH \
    --shuffle \
    $DATA_ROOT \
    $DATA/newtest.txt \
    $EXAMPLE/bacteria_val_lmdb
   #$DATA/purelabel.txt \
    #$EXAMPLE/bacteria_label_lmdb

echo "Done."
