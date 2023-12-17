import nn
from functions import *
from layer import *
from manager import *


import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('Iris.csv')

numeric_data = df[df.columns[[1,2,3,4]]]
categorial_data = df[df.columns[5]]

dummy_features = pd.get_dummies(categorial_data)

data = pd.concat([numeric_data, dummy_features], axis=1)

X = data[data.columns[[0,1,2,3]]]
y = data[data.columns[[4,5,6]]]

x_train, x_test, y_train, y_test = train_test_split(X.values, y.values, train_size=0.8, random_state=42)

md = nn.NeuralNetwork()

md.add_layers(input_layer('input', 4))

md.add_layers(hide_layer('hide', 10,sigmoida()))

md.add_layers(output_layer('output', 3, softmax()))

md.compile()



manag = manager(md, 0.01, SoftMaxCrossEntropy())
manag.fit(x_train,y_train,x_test, y_test, 1000)



print('1 0 0 - setosa; 0 1 0 - versicolor; 0 0 1 verginica')
print(md.forward([4.8, 3.1, 1.6, 0.2])) # setosa
print(md.forward([5.6, 3.0, 4.1, 1.3])) # versicolor
print(md.forward([5.9, 3.0, 5.1, 1.8])) #verginica








