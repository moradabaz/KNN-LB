import random
import sys
import time
import timeit
from datetime import date
import pandas  as pd
from sktime import classification
from sktime.classification.distance_based import KNeighborsTimeSeriesClassifier
from sktime.utils.load_data import load_from_tsfile_to_dataframe as ts_loader
from sktime.utils.load_data import load_from_arff_to_dataframe as arff_loader
import numpy as np
from FileReader import FileReader
from Sequence_stats import SequenceStats

knn = KNeighborsTimeSeriesClassifier()

name = ""
training_path = ""
testing_path = ""
window = 0
D = 0
V = 0
neighbors = 1
random.seed(1234)

if len(sys.argv) > 1:
    for i in range(2, len(sys.argv)):
        options = sys.argv[i].split("=")
        arg = options[0]
        value = options[1]
        if arg == "-name":
            name = value
        elif arg == "-train":
            training_path = value
        elif arg == "-test":
            testing_path = value
        elif arg == "-window":
            window = float(value)
        elif arg == "-n":
            neighbors = int(value)
        elif arg == "-v":
            V = float(value)

# Load data
train_file = FileReader.load_data(training_path)
test_file = FileReader.load_data(testing_path)


data_train = arff_loader(training_path)
data_test = arff_loader(testing_path)



# Create datasets

train_data, train_labels = FileReader.parse_arff_data(train_file)
test_data, test_labels = FileReader.parse_arff_data(test_file)

train_cache = SequenceStats(train_data)
test_cache = SequenceStats(test_data)

train_data = np.array(train_data)
train_labels = np.array(train_labels)
test_data = np.array(test_data)
test_labels = np.array(test_labels)

knn = KNeighborsTimeSeriesClassifier()
knn.fit(data_train[0], data_train[1])
predictions = knn.score(data_test[0], data_test[1])
print(predictions)

"""
#knn = KnnLb.KnnDtw(n_neighbors=neighbors, max_warping_window=10)
df = pd.DataFrame(train_data,index=train_data[:,0])
knn.fit(df, train_labels)
start = timeit.default_timer()
label = knn.predict(test_data)
print(label)
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

f_path = '../outputs/' + name + '_KNN_LB_' + str(date.today()) + "_" + \
         str(time.localtime().tm_hour) + "-" + str(time.localtime().tm_min) + "-" + \
         str(time.localtime().tm_sec) + ".csv"
linea = name + ',' + str(window) + ',' + str(V) + ',' + str(neighbors) + ',' + str(round(accuracy, 5)) + ',' + str(round(exec_time, 5))
with open(f_path, 'w+') as file:
    file.writelines("name,window,V,Neighbors,accuracy,exec_time\n")
    file.write("%s\n" % linea)
file.close()

"""

