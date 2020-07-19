#!/bin/bash
CUR_DIR=$(cd .. && pwd)
for d in $CUR_DIR/datasets/*; do
  if [ -d $d ]; then
    dataset=$(basename $d)
    DATASET_DIR=$CUR_DIR/datasets/$dataset
    echo ""
    for neighbors in {1..10}
    do
      for window in {1..10}
      do
        for v in {1..10}
        do
          echo "Executing dataset [ $dataset ]  ..."
          python3 ../Runner.py $CUR_DIR -name=$dataset -train=$DATASET_DIR/"$dataset"_TRAIN.arff -test=$DATASET_DIR/"$dataset"_TEST.arff -n=$neighbors -window=$window -v=$v
          echo "Classification process [ $dataset ] finished"
        done
      done
    done
  fi
done

