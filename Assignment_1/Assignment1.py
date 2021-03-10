from numpy import *
import pandas as pd

#Read text file
df = pd.read_csv('train-a1-449.txt', sep=" ", header=None)

#Replace Y and N with -1 and 1
df[1024].replace({"N": -1, "Y": 1}, inplace=True)

#Create y and w
y = df[1024]
del df[1024]
del df[1025]
x = df
w = zeros((1024,))
p = zeros((792,))
m = zeros(792,)
dim = x.shape

#Normalize
x = x.div(x.max().max())

#Sign function that returns sign 0 as 1, not 0
def my_sign(x):
    if sign(x) == 0:
        return 1
    else:
        return sign(x)

#Train classifier
while sum(p != y) != 0:
    for i in range(dim[0]):
        p[i] = my_sign(dot(x.iloc[i].transpose(), w))
        if(p[i] != y[i]):
            w += y[i]*x.iloc[i]

#Calculate margin
for i in range(dim[0]):
    m[i] = linalg.norm(x.iloc[i] - w)

print(min(m))
savetxt("classifier.txt", w, delimiter = " ")
