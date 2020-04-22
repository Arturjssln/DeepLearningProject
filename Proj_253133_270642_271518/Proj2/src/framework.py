from torch import empty
import math

class Module(object):
    autograd = True
    def __init__(self):
        raise NotImplementedError

    def __call__(self, *input):
        return self.forward(input)

    def forward(self, *input):
        raise NotImplementedError

    def backward(self, *gradwrtoutput):
        raise NotImplementedError

    def param(self):
        return []

    def criterion(self):
        raise NotImplementedError

class zero_grad(Module):
    raise NotImplementedError

class no_grad(Module):
    def __enter__(self):
        Module.autograd = False

    def __exit__(self, type, value, traceback):
        Module.autograd = True


class Linear(Module):
    raise NotImplementedError

class ReLU(Module):
    def __call__(self, *input):
        return self.forward(input)

    def forward(self, *input):
        raise NotImplementedError
        if Module.autograd:
            return input if input > 0 else 0
        else:
            return input if input > 0 else 0

class Tanh(Module):
    def __call__(self, *input):
        return self.forward(input)

    def forward(self, *input):
        raise NotImplementedError
        if Module.autograd:
            #Add gradient graph
            return math.tanh(input)
        else:
            #Add gradient graph
            return math.tanh(input)

class Sequential(Module):
    raise NotImplementedError

class LossMSE(Module):
    raise NotImplementedError
