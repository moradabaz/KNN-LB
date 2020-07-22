#!/bin/bash
CUR_DIR=$(cd .. && pwd)
for d in $CUR_DIR/datasets/*; do
  if [ -d $d ]; then
    dataset=$(basename $d)
    DATASET_DIR=$CUR_DIR/datasets/$dataset
    echo ""
    echo "Executing dataset [ $dataset ]  ..."
    python3 ../RunnerDTW.py $CUR_DIR -name=$dataset -train=$DATASET_DIR/"$dataset"_TRAIN.arff -test=$DATASET_DIR/"$dataset"_TEST.arff -n=1 -window=1 -v=20
    echo "Classification process [ $dataset ] finished"
  fi
done
