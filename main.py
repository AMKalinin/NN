import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split


import nn
from functions import *
from layer import *
from manager import *

df = pd.read_csv('mnist_train.csv', header=None)


num = range(1,785)
numeric_data = df[df.columns[num]]
categorial_data = df[df.columns[0]]

dummy_features = pd.get_dummies(categorial_data)


num = range(0,784)
x = numeric_data[numeric_data.columns[num]]
y = dummy_features[dummy_features.columns[[0,1,2,3,4,5,6,7,8,9]]]


x_train, x_test, y_train, y_test = train_test_split(x.values, y.values, train_size=0.8, random_state=42)


md = nn.NeuralNetwork()

md.add_layers(input_layer( 'input', 784))
# md.add_layers(hide_layer('hide', 512, sigmoida()))
# md.add_layers(hide_layer('hide', 128, sigmoida()))
# md.add_layers(hide_layer('hide', 32, sigmoida()))
md.add_layers(output_layer('output', 10, softmax()))

md.compile()


# md = nn.NeuralNetwork()

# md.add_layers(input_layer( 'input', 3))

# md.add_layers(output_layer('output', 1, softmax()))

# md.compile()

# md.layers[1].neurons[0].w = [-0.16595599, 0.44064899,-0.99977125]   


manag = manager(md, 0.01, SoftMaxCrossEntropy())



# x_train = [[0,0,1],
#             [1,1,1],
#             [1,0,1],
#             [0,1,1]]

# y_train = [[0],[1],[1],[0]]
# y_train = np.array(y_train)
# x_test = [[1,0,0]]
# y_test = [[1]]

# y_test = np.array(y_test)


manag.fit(x_train/255,y_train,x_test/255, y_test, 10)


print(md.forward(x_train[0]/255))
print(md.forward(x_train[1]/255))
print(md.forward(x_train[3]/255))


