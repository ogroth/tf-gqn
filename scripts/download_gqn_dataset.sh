#!/bin/bash

# Downloads the GQN dataset in parts from the Google cloud bucket.
# $1: dataset part
# $2: target directory

# dataset description
GQN_DATASET_NAME="gqn-dataset"
GQN_DATASET_ROOT_URL="https://console.cloud.google.com/storage/browser/gqn-dataset"
declare -a GQN_DATASET_PARTS=(
  "jaco"
  "mazes"
  "rooms_free_camera_no_object_rotations"
  "rooms_free_camera_with_object_rotations"
  "rooms_ring_camera"
  "shepard_metzler_5_parts"
  "shepard_metzler_7_parts"
)
declare -a GQN_DATASET_TRAIN_SIZES=(
  3600
  1080
  2160
  2034
  2160
  900
  900
)
declare -a GQN_DATASET_TEST_SIZES=(
  400
  120
  240
  226
  240
  100
  100
)

# parse parameters
DATASET_PART=$1
DATA_DIR=$2

# get array index
for i in "${!GQN_DATASET_PARTS[@]}"; do
  if [[ "${GQN_DATASET_PARTS[$i]}" = "${DATASET_PART}" ]]; then
    PART_IDX="${i}"
  fi
done

# create destination directory
DST_DIR=${DATA_DIR}/${GQN_DATASET_NAME}/${DATASET_PART}
mkdir -p ${DST_DIR}

# download train data
mkdir -p ${DST_DIR}/train
max_idx="${GQN_DATASET_TRAIN_SIZES[${PART_IDX}]}"
for i in `seq 1 ${max_idx}`; do
  idx=`printf '%03d' ${i}`
  fn=${idx}-of-${max_idx}.tfrecord
  wget ${GQN_DATASET_ROOT_URL}/${DATASET_PART}/train/${fn} -O ${DST_DIR}/train/${fn}
done

# download test data
mkdir -p ${DST_DIR}/test
max_idx="${GQN_DATASET_TEST_SIZES[${PART_IDX}]}"
for i in `seq 1 ${max_idx}`; do
  idx=`printf '%03d' ${i}`
  fn=${idx}-of-${max_idx}.tfrecord
  wget ${GQN_DATASET_ROOT_URL}/${DATASET_PART}/test/${fn} -O ${DST_DIR}/test/${fn}
done
