'''Модуль с функциями активации,производными и функциями потерь'''

from math import exp
from math import log


class function_activation():
    def activate(self):
        pass

    def derivative(self):
        pass

class ReLU(function_activation):
    def __init__(self):
        self.type = 'RELU'

    def activate(self, neu):
        if( neu.sum < 0):
            neu.out = 0
        else:
            neu.out = neu.sum
    
    def derivative(self, neu):
        if( neu.sum < 0):
            return 0
        else:
            return 1

class sigmoida(function_activation):
    def __init__(self):
        self.type = 'sigmoida'
    
    def activate(self, neu):
        neu.out = 1/(1+exp(-neu.sum))
    
    def derivative(self, neu):
        return neu.out*(1-neu.out)



class tanh(function_activation):

    def __init__(self):
        self.type = 'tanh'
    
    def activate(self, neu):
        a = exp(neu.sum)
        b =  exp(-neu.sum)
        neu.out = (a - b) / (a + b)
    
    def derivative(self, neu):
        return (1-neu.out*neu.out)


class softmax(function_activation):
    def __init__(self):
        self.type = 'softmax'
    
    def activate(self, neu):
        pass
    
    def derivative(self, neu):
        pass


class nonFunc(function_activation):
    def __init__(self) -> None:
        self.type = 'nonFunctions'

    def activate(self, neu):
        neu.out = neu.sum
    
    def derivative(self, neu):
        return 1  


class Loss:
    def loss(self, predict, target):
        pass

    def gradient(self, predict, target):
        pass

class SSE(Loss):
    def loss(self, predict, target):
        squared_errors = 0
        for i in range(len(predict)):
            squared_errors += (predict[i]-target[i])**2
        return squared_errors

    def gradient(self, predict, target):
        grad = []
        for i in range(len(predict)):
            grad.append(2*(predict[i]-target[i]))
        return grad


class SoftMaxCrossEntropy(Loss):
    def loss(self, predict, target):
        loss = 0
        for i in range(len(predict)):
            loss += log(predict[i] + 1e-30)*target[i]
        return -loss

    def gradient(self, predict, target):
        grad = []
        for i in range(len(predict)):
            grad.append(predict[i]-target[i])
        return grad
