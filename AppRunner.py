from numpy import double
from sklearn.metrics import classification_report, confusion_matrix
import KnnLb
import sys
import numpy as np
import time
import Sequence_stats
from   FileReader import FileReader
import pandas as pd

# Import the HAR dataset
train_file = FileReader.load_data('/Users/morad/PycharmProjects/KNN-LB/datasets/ItalyPowerDemand'
                                    '/ItalyPowerDemand_TRAIN.arff')
test_file = FileReader.load_data('/Users/morad/PycharmProjects/KNN-LB/datasets/ItalyPowerDemand'
                                    '/ItalyPowerDemand_TEST.arff' )

# Create empty lists
train_data, train_labels = FileReader.parse_arff_data(train_file)
test_data, test_labels = FileReader.parse_arff_data(test_file)

train_cache = Sequence_stats.SequenceStats(train_data)
test_cache = Sequence_stats.SequenceStats(test_data)

train_data = np.array(train_data)
train_labels = np.array(train_labels)
test_data = np.array(test_data)
test_labels = np.array(test_labels)
#train_data.data
m = KnnLb.KnnDtw(n_neighbors=1, max_warping_window=10)
m.fit(train_data, train_labels)
label, proba = m.predict_lb(test_data, test_cache)

aciertos = 0
fallos = 0
tam_labels = len(test_labels)
for i in range(0, len(test_labels)):
    if label[i] == test_labels[i]:
        aciertos = aciertos + 1
    else:
        fallos = fallos + 1

print("Accuracy: ", aciertos / len(test_labels))
