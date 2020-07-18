from sklearn.metrics import classification_report
from dtaidistance import dtw
from scipy.io import arff
from FileReader import FileReader as filereader
import numpy as np

train_path = '/Users/morad/PycharmProjects/KNN-LB/datasets/ItalyPowerDemand/ItalyPowerDemand_TRAIN.arff'
test_path = '/Users/morad/PycharmProjects/KNN-LB/datasets/ItalyPowerDemand/ItalyPowerDemand_TEST.arff'


def knn(train, test, w):
    preds = []
    for ind, i in enumerate(test):
        min_dist = float('inf')
        closest_seq = []
        for j in train:
            # if LB_Keogh(i[:-1], j[:-1], 5) < min_dist:
            dist = dtw.distance_fast(i[:-1], j[:-1], window=w)
            if dist < min_dist:
                min_dist = dist
                closest_seq = j
        preds.append(closest_seq[-1])
    return classification_report(test[:, -1], preds)


def knn_alt(train, test, w):
    preds = []
    for series_test in test:
        min_dist = float('inf')
        closest_seq = []
        for serie_train in train:
            dist = dtw.distance_fast(series_test, serie_train, window=w)
            if dist < min_dist:
                min_dist = dist
                closest_seq = serie_train
        preds.append(closest_seq)
    for pred in preds:
        print(pred)
    return classification_report(test, preds)


pass

f_train = open(train_path)
f_test = open(test_path)
data_train = arff.loadarff(f_train)
data_test = arff.loadarff(f_test)
dt_train, train_labels = filereader.load_arff_data(train_path)
dt_test, test_labels = filereader.load_arff_data(test_path)
classification = knn_alt(dt_train, dt_test, 0.2)
print(classification)
