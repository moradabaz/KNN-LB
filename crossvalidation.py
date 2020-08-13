import sys
import json
import time
import random
import timeit
import numpy as np
sys.path.append(sys.argv[1])
from Sequence_stats import SequenceStats
import KnnLb
from FileReader import FileReader

trainin_path = sys.argv[3]
name = sys.argv[2]
window = float(sys.argv[4])
V = float(sys.argv[5])
neighbors = int(sys.argv[6])
random.seed(1234)

train_file = FileReader.load_data(trainin_path)
num_series = len(train_file)
num_series_per_fold = round(num_series / 10)
counter = 0
folds = dict()
key = 0
series = list()
for i in range(0, num_series):
    if i % num_series_per_fold is 0:
        if key < 10:
            key = key + 1
            folds[key] = list()
            folds[key].append(train_file[i])
        else:
            folds[key].append(train_file[i])
    else:
        folds[key].append(train_file[i])

train_data = list()
label_data = list()
fold_accuracies = dict()
resultados = list()
for turn in folds.keys():
    test_dataset, test_labels = FileReader.parse_arff_data(folds[turn])
    for fold in folds.keys():
        if fold != turn:
            for line in folds[fold]:
                train_data.append(line)
    print("FOLD ", turn, " ->")
    train_dataset, train_labels = FileReader.parse_arff_data(train_data)
    test_cache = SequenceStats(test_dataset)
    train_dataset, train_labels = np.asarray(train_dataset), np.asarray(train_labels)
    test_dataset, test_labels = np.asarray(test_dataset), np.asarray(test_labels)
    window = 2
    m = KnnLb.KnnDtw(n_neighbors=neighbors, max_warping_window=window)
    m.fit(train_dataset, train_labels)
    start = timeit.default_timer()
    label, proba = m.predict_lb(test_dataset, test_cache, window, V)
    stop = timeit.default_timer()

    aciertos = 0
    fallos = 0
    tam_labels = len(test_labels)
    for i in range(0, len(test_labels)):
        if label[i] == test_labels[i]:
            aciertos = aciertos + 1
        else:
            fallos = fallos + 1

    accuracy = aciertos / len(test_labels)
    exec_time = (stop - start)
    print("[ACCURACY]: ", accuracy)
    print("Time execution: ", exec_time)
    linea = str(turn) + ',' + str(window) + ',' + str(V) + ',' + str(neighbors) + ',' + str(round(accuracy, 5)) + ',' + str(round(exec_time, 5))
    resultados.append(linea)

f_path = '../outputs/KNN_' + name + '_Folds_' + str(time.localtime().tm_hour) + str(time.localtime().tm_min) + str(
        time.localtime().tm_sec) + ".csv"
with open(f_path, 'w+') as file:
    file.writelines("iterations,window,V,Neighbors,accuracy,exec_time\n")
    file.writelines("%s\n" % linea for linea in resultados)
file.close()
