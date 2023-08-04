import numpy as np

def sigmoidFunc(value):
    return 1 / (1 + np.exp(-value))

def sigmoidPrime(value):
    activation = sigmoidFunc(value)
    return activation * (1 - activation)

def ReLuFunc(value):
    return np.maximum(0.1 * value, value)

def ReLuPrime(value):
    return ReLuFunc(value) / value

def GeLuFunc(value):
    return value * sigmoidFunc(value)

def GeLuPrime(value):
    return sigmoidFunc(value) + value * sigmoidPrime(value)

def linearFunc(value):
    return value

def linearPrime(value):
    m = np.zeros(value.shape)
    m.fill(1.0)
    return m

activationDic = {
    'Sigmoid' : (sigmoidFunc, sigmoidPrime),
    'GeLu' : (GeLuFunc, GeLuPrime),
    'ReLu' : (ReLuFunc, ReLuPrime),
    'Linear' : (linearFunc, linearPrime)
}