#Assignment 2

import numpy as np
import pandas as pd

training_data = pd.read_csv('a2-train-data.txt', sep=" ", header=None)
training_labels = pd.read_csv('a2-train-label.txt', sep=" ", header=None)
testing_data = pd.read_csv('a2-test-data.txt', sep=" ", header=None)
testing_labels = pd.read_csv('a2-test-label.txt', sep=" ", header=None)

del training_data[1000]

num_units = 10

#Used: https://pylessons.com/Logistic-Regression-part1/
def sigmoid(x):
    s = 1/(1+np.exp(-x))
    return s

def sigmoid_derivative(x):
    s = sigmoid(x)
    ds = s*(1-s)
    return ds

def my_sign(x):
    return [1 if  i >= 0 else -1 for i in x]

#Heavily used
#https://towardsdatascience.com/how-to-build-a-simple-neural-network-from-scratch-with-python-9f011896d2f3
class NeuralNetwork:
    def __init__(self, x, y):
        self.input      = x
        self.w1   = np.random.rand(self.input.shape[1],num_units)
        self.w2   = np.random.rand(num_units,1)
        self.y          = y
        self.output     = np.zeros(y.shape)

    def feedforward(self):
        self.layer1 = sigmoid(np.dot(self.input, self.w1))
        self.output = sigmoid(np.dot(self.layer1, self.w2))
        return self.output

    def backprop(self):
        d_w2 = np.dot(self.layer1.T, (2*(self.y - self.output) * sigmoid_derivative(self.output)))
        d_w1 = np.dot(self.input.T,  (np.dot(2*(self.y - self.output) * sigmoid_derivative(self.output), self.w2.T) * sigmoid_derivative(self.layer1)))
        self.w1 += d_w1
        self.w2 += d_w2

    def train(self, epochs):
        for j in range(epochs):
                self.feedforward()
                self.backprop()

    def predict(self, x):
        self.input = x
        ff = self.feedforward()
        preds = [1 if  i >= 0.5 else -1 for i in ff]
        return np.array(preds)

    def return_guts(self):
        return(self.w1, self.w2)

NN = NeuralNetwork(training_data, training_labels)
NN.train(5000)
predstrain = NN.predict(training_data)
predstest = NN.predict(testing_data)

#Manipulate test/train label data structure
testing_labels = np.array(testing_labels.stack())
training_labels = training_labels[0]


w1, w2 = NN.return_guts()

print(w1.shape )
print(w2.shape)

print("Training accuracy",sum(predstrain == training_labels)/training_labels.size*100)
print("Testing accuracy", sum(predstest == testing_labels)/testing_labels.size*100)

with open("predictions.txt", "w") as fout1:
    np.savetxt(fout1, predstrain, delimiter = " " )

with open("neuralnet.txt", "w") as fout2:
    fout2.write("10\n")
    np.savetxt(fout2, NN.w2, delimiter = " ")
    fout2.write("\n\n")
    np.savetxt(fout2, NN.w1, delimiter = " ")
