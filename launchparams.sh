#!/bin/bash
CUR_DIR=$(cd .. && pwd)
for d in $CUR_DIR/datasets/*; do
  if [ -d $d ]; then
    dataset=$(basename $d)
    DATASET_DIR=$CUR_DIR/datasets/$dataset
    echo ""
    for window in 1 2 3 4 5
    do
      for v in 1 2.2 3.2 4.2 5.2
      do
        echo "Executing dataset [ $dataset ]  ..."
        python3 ../crossvalidation.py $CUR_DIR $dataset $DATASET_DIR/"$dataset"_TRAIN.arff $window $v 1
        echo "Classification process [ $dataset ] finished"
      done
    done
  fi
done
