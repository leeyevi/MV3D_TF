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
NET=$3
DATASET=$4

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:4:$len}
EXTRA_ARGS_SLUG=${EXTRA_ARGS// /_}



LOG="experiments/logs/mv3d_end2end_${NET}_${EXTRA_ARGS_SLUG}.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

time python ./tools/train_net.py --device ${DEV} --device_id ${DEV_ID} \
  --weights data/pretrain_model/VGG_imagenet.npy \
  --imdb 'kitti_train' \
  --iters 10000 \
  --cfg experiments/cfgs/faster_rcnn_end2end.yml \
  --network MV3D_train \
  ${EXTRA_ARGS}

set +x
NET_FINAL=`grep -B 1 "done solving" ${LOG} | grep "Wrote snapshot" | awk '{print $4}'`
set -x

# time python ./tools/test_net.py --device ${DEV} --device_id ${DEV_ID} \
  # --weights /home/radmin/code/Faster-RCNN_TF/output/faster_rcnn_end2end/train/VGGnet_fast_rcnn_iter_14000.ckpt.meta \
  # --imdb kitti_test \
  # --cfg experiments/cfgs/faster_rcnn_end2end.yml \
  # --network MV3D_test \
  # ${EXTRA_ARGS}
