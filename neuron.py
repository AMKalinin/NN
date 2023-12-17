'''Модуль для описания нейрона входящего в состав слоя'''

from functions import *


class neuron:
    def __init__(self):
        self.out = 0
    
    def activate(self):
        pass

class input_neuron(neuron):
    
    def __init__(self):
        super().__init__()

    def activate(self):
        pass


class hide_neuron(neuron):

    def __init__(self, b =0, w = None, func_activation = None):
        super().__init__()
        self.b = b
        self.w = w
        self.sum = 0
        self.func = func_activation


    def activate(self):
        self.func.activate(self)

    def derivative(self):
        return self.func.derivative(self)


class output_neuron(hide_neuron):

    def __init__(self, b=0, w=None, func_activation=None):
        super().__init__(b=b, w=w, func_activation=func_activation)

    def activate(self):
        self.func.activate(self)

    def derivative(self):
        return self.func.derivative(self)
