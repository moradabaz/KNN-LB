import os

from sklearn.metrics import classification_report
from dtaidistance import dtw
from scipy.io import arff
from FileReader import FileReader as filereader
import numpy as np

train_path = '/Users/morad/PycharmProjects/KNN-LB/datasets/ItalyPowerDemand/ItalyPowerDemand_TRAIN.arff'
test_path = '/Users/morad/PycharmProjects/KNN-LB/datasets/ItalyPowerDemand/ItalyPowerDemand_TEST.arff'

d = './datasets'
var = [os.path.join(d, o) for o in os.listdir(d)
       if os.path.isdir(os.path.join(d, o))]

datasets = list()
for dt in var:
    size = len(dt.split('/'))
    # dt.split('/')[size - 1]
    datasets.append(dt.split('/')[size - 1])

print(datasets)


print(os.getcwd())