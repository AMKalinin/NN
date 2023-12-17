import nn
from functions import *
from layer import *
from manager import *

import math
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

x = -math.pi
xds = []
yds = []
x_l = []
y_l = []
while x<math.pi:
    xds.append([x])
    x_l.append(x)
    a = math.sin(x)
    yds.append([a])
    y_l.append(a)
    x += 0.001


x_train, x_test, y_train, y_test = train_test_split(xds, yds, train_size=0.8, random_state=42)



md = nn.NeuralNetwork()
md.add_layers(input_layer( 'input', 1))
md.add_layers(hide_layer('hide', 10, sigmoida()))
md.add_layers(output_layer('output', 1, nonFunc()))
md.compile()

manag = manager(md, 0.01, SSE())
manag.fit(x_train,y_train,x_test, y_test, 100)

print(md.forward([0]))
predict_func = []

for i in range(len(x_l)):
    predict_func.append(md.forward(xds[i]))


plt.plot(x_l, predict_func, x_l,y_l)
plt.show()