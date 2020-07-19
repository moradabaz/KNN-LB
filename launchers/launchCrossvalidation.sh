CUR_DIR=$(cd .. && pwd)
DATASET_DIR=$CUR_DIR/datasets/$1/
python3 ../crossvalidation.py $CUR_DIR $1 $DATASET_DIR/$1_TRAIN.arff -window=2 -d=2.4 -v=4.2


#!/bin/bash
CUR_DIR=$(cd .. && pwd)
for d in $CUR_DIR/datasets/*; do
  if [ -d $d ]; then
    dataset=$(basename $d)
    DATASET_DIR=$CUR_DIR/datasets/$dataset
    echo ""
    for neighbors in 1 2 3 4 5 6 7 8 9 10
    do
      for window in 1 2 3 4 5 6 7 8 9 10
      do
        for v in 1 2 3 4 5 6 7 8 9 10
        do
          echo "Executing dataset [ $dataset ]  ..."
          python3 ../crossvalidation.py $CUR_DIR -name=$dataset -train=$DATASET_DIR/"$dataset"_TRAIN.arff $window $v $neighbors
          echo "Classification process [ $dataset ] finished"
        done
      done
    done
  fi
done
