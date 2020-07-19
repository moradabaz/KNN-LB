CUR_DIR=$(cd .. && pwd)
DATASET_DIR=$CUR_DIR/datasets/$1/
python3 ../crossvalidation.py $CUR_DIR $1 $DATASET_DIR/$1_TRAIN.arff -window=2 -d=2.4 -v=4.2