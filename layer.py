from neuron import *

class layer:
    def __init__(self, type, size):
        self.size = size
        self.neurons = []
        self.type = type


class input_layer(layer):
    def __init__(self, type, size):
        super().__init__(type, size)
        self.add_neurons()

    def add_neurons(self):
        for i in range(self.size):
            self.neurons.append(neuron())
        
        
class hide_layer(layer):
    def __init__(self, type, size, func_activation):
        super().__init__(type, size)
        self.add_neurons(func_activation)
    
    def add_neurons(self, func):
        for i in range(self.size):
            self.neurons.append(hide_neuron(func_activation=func))

    def activate(self):
        for i in range(self.size):
            self.neurons[i].activate()

    def derivative(self):
        der = []
        for i in range(self.size):
            der.append(self.neurons[i].derivative())
        return der
    


class output_layer(layer):
    def __init__(self, type, size, func_activation):
        super().__init__(type, size)
        self.add_neurons(func_activation)
        self.func_type = func_activation.type

    def add_neurons(self, func):
        if func.type != 'softmax':
            for i in range(self.size):
                self.neurons.append(output_neuron(func_activation=func))
        else:
            for i in range(self.size):
                self.neurons.append(output_neuron())

    def activate(self):
        if self.func_type != 'softmax':
            for i in range(self.size):
                self.neurons[i].activate()
        else:
            sum = 0
            max = self.neurons[0].sum
            for i in range(1,self.size):
                if (max < self.neurons[i].sum):
                    max = self.neurons[i].sum
            for i in range(self.size):
                sum += exp(self.neurons[i].sum-max)
            for i in range(self.size):
                self.neurons[i].out = exp(self.neurons[i].sum)/sum

    def derivative(self):
        der = []
        if self.func_type != 'softmax':
            for i in range(self.size):
                der.append(self.neurons[i].derivative())
        else:
            for i in range(self.size):
                der.append(1)
        return der
                
        

        







