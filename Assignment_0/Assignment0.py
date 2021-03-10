from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import pandas as pd

training_data = pd.read_csv('a2-train-data.txt', sep=" ", header=None)
training_labels = pd.read_csv('a2-train-label.txt', sep=" ", header=None)
testing_data = pd.read_csv('a2-test-data.txt', sep=" ", header=None)
testing_labels = pd.read_csv('a2-test-label.txt', sep=" ", header=None)

del training_data[1000]

##scikitlearn
NN = MLPClassifier((25, 25, 25), max_iter=5000)
NN.fit(training_data, training_labels.values.ravel())
predictions = NN.predict(testing_data)
print(classification_report(testing_labels,predictions))


#Tensorflow
model = Sequential()
model.add(Dense(12, input_dim=1000, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(training_data, training_labels, epochs=150, batch_size=10)

model.predict(testing_labels)
