#!/bin/bash
# Usage:
# ./experiments/scripts/faster_rcnn_end2end.sh GPU NET DATASET [options args to {train,test}_net.py]
# DATASET is either pascal_voc or coco.
#
# Example:
# ./experiments/scripts/faster_rcnn_end2end.sh 0 VGG_CNN_M_1024 pascal_voc \
#   --set EXP_DIR foobar RNG_SEED 42 TRAIN.SCALES "[400, 500, 600, 700]"

set -x
set -e

export PYTHONUNBUFFERED="True"

DEV=$1
DEV_ID=$2
WEIGHTS=$3
DATASET=$4

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:4:$len}
EXTRA_ARGS_SLUG=${EXTRA_ARGS// /_}



LOG="experiments/logs/mv3d_end2end_.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

# time python ./tools/train_net.py --device ${DEV} --device_id ${DEV_ID} \
  # --weights ${WEIGHTS}\
  # --imdb ${DATASET} \
  # --iters 10001\
  # --cfg experiments/cfgs/faster_rcnn_end2end.yml \
  # --network MV3D_train \
  # ${EXTRA_ARGS}

  # --weights data/pretrain_model/vgg_imagenet_sampled.npy \
set +x
NET_FINAL=`grep -B 1 "done solving" ${LOG} | grep "Wrote snapshot" | awk '{print $4}'`
set -x

time python ./tools/test_net.py --device ${DEV} --device_id ${DEV_ID} \
  --weights ${WEIGHTS} \
  --imdb ${DATASET}\
  --cfg experiments/cfgs/faster_rcnn_end2end.yml \
  --network MV3D_test \
  ${EXTRA_ARGS}
