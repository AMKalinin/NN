from random import random

class model():
    def __init__(self):
        pass
    
    def add_layers(self):
        pass
    
    def compile(self):
        pass

    def forward(self):
        pass
    
    def backward(self):
        pass

    def fit(self):
        pass

    def init_w(self):
        pass


class NeuralNetwork(model):

    def __init__(self):
        self.layers = []
        self.lr = 1

    def add_layers(self, layer):
        self.layers.append(layer)

    def init_w(self, size):
        w = []
        for i in range(size):
            w.append(random()-0.5)
        return w
        

    def compile(self):
        self.size = len(self.layers)
        for i in range(1,self.size):
            for neuron in self.layers[i].neurons:
                neuron.w = self.init_w(self.layers[i-1].size)

    def forward(self, X):
        out = []
        for i in range(len(X)):
            self.layers[0].neurons[i].out = X[i]
        for i in range(1, self.size):
            for j in range(self.layers[i].size):
                self.layers[i].neurons[j].sum = 0
                for k in range(self.layers[i-1].size):
                    self.layers[i].neurons[j].sum += self.layers[i-1].neurons[k].out * self.layers[i].neurons[j].w[k] 
                self.layers[i].neurons[j].sum += self.layers[i].neurons[j].b      
            self.layers[i].activate()
        for i in range(self.layers[-1].size):
            out.append(self.layers[-1].neurons[i].out)
        return out
            
    def backward(self, gradient):
        for k in reversed(range(len(self.layers)-1)):
            gradientNext = []
            der = self.layers[k+1].derivative()
            for i in range(self.layers[k+1].size):
                    gradient[i] *= der[i] * self.lr
            for i in range(self.layers[k].size):
                gradientNext.append(0)
                for j in range(self.layers[k+1].size):
                    gradientNext[i] += self.layers[k+1].neurons[j].w[i] * gradient[j]
            
            for i in range(self.layers[k+1].size):
                for j in range(self.layers[k].size):
                    self.layers[k+1].neurons[i].w[j] -= gradient[i] * self.layers[k].neurons[j].out
                self.layers[k+1].neurons[i].b -= gradient[i]

            gradient = gradientNext.copy()



class NN_momentum(NeuralNetwork):
    def __init__(self):
        super().__init__()

    def backward(self, gradient):
        pass
