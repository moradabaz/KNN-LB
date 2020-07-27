CUR_DIR=$(cd .. && pwd)
DATASET_DIR=$CUR_DIR/datasets/$1/
python3 ../Runner.py $CUR_DIR -name=$1 -train=$DATASET_DIR/$1_TRAIN.arff -test=$DATASET_DIR/$1_TEST.arff -n=1 -window=1 -v=10