#!/bin/bash
GQN_DATASET_ROOT_URL="https://console.cloud.google.com/storage/browser/gqn-dataset"

DATA_DIR=${HOME}/data/
DST_DIR=${DATA_DIR}/gqn-dataset/shepard_metzler_5_parts/
mkdir -p ${DST_DIR}

# train data
mkdir -p ${DST_DIR}/train
for i in `seq 1 900`; do
  idx=`printf '%03d' ${i}`
  fn=${idx}-of-900.tfrecord
  wget ${GQN_DATASET_ROOT_URL}/shepard_metzler_5_parts/train/${fn} -O ${DST_DIR}/train/${fn}
  done
