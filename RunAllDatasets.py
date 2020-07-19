import os
import sys
import timeit
sys.path.append(sys.argv[1])
from datetime import date
import numpy as np
import KnnLb
from FileReader import FileReader
from Sequence_stats import SequenceStats
import time


window=0
V=0

d = './datasets'
var = [os.path.join(d, o) for o in os.listdir(d)
       if os.path.isdir(os.path.join(d, o))]

datasets = list()
for dt in var:
    size = len(dt.split('/'))
    datasets.append(str(dt.split('/')[size - 1]))


if len(sys.argv) > 1:
    for i in range(2, len(sys.argv)):
        options = sys.argv[i].split("=")
        arg = options[0]
        value = options[1]
        if arg == "-window":
            window = float(value)
        elif arg == "-v":
            V = float(value)

current_dir = os.getcwd() + '/datasets'
resultados = list()
for dt_name in datasets:
    name = dt_name
    training_path = current_dir + '/' + str(dt_name) + '/' +  str(dt_name) + '_TRAIN.arff'
    testing_path = current_dir + '/' + str(dt_name) + '/' +  str(dt_name) + '_TRAIN.arff'

    # Load data
    train_file = FileReader.load_data(training_path)
    test_file = FileReader.load_data(testing_path)

    # Create datasets

    train_data, train_labels = FileReader.parse_arff_data(train_file)
    test_data, test_labels = FileReader.parse_arff_data(test_file)

    train_cache = SequenceStats(train_data)
    test_cache = SequenceStats(test_data)

    train_data = np.array(train_data)
    train_labels = np.array(train_labels)
    test_data = np.array(test_data)
    test_labels = np.array(test_labels)

    m = KnnLb.KnnDtw(n_neighbors=1, max_warping_window=10)
    m.fit(train_data, train_labels)
    start = timeit.default_timer()
    label, proba = m.predict_lb(test_data, test_cache, window, V)
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
    accuracy = round(accuracy, 5)
    exec_time = (stop - start)
    exec_time = round(exec_time, 5)
    print("Accuracy: ", accuracy)
    print("Time execution: ", exec_time)
    linea = name + ',' + str(window) + ',' + str(V) + ',' + str(round(accuracy, 5)) + ',' + str(round(exec_time, 5))
    resultados.append(linea)
f_path = '../outputs/All_KNN_LB_' + str(date.today()) + "_" + \
             str(time.localtime().tm_hour) + "-" + str(time.localtime().tm_min) + "-" + \
             str(time.localtime().tm_sec) + ".csv"


with open(f_path, 'w+') as file:
    file.writelines("name,window,V,accuracy,exec_time\n")
    file.writelines("%s\n" % linea for linea in resultados)
file.close()
