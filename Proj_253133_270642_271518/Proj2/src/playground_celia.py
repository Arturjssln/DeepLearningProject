from torch import empty
import math

class Module(object):
    autograd = True #why?
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
        #when no_grad called: autograd desactivated
        Module.autograd = False

    def __exit__(self, type, value, traceback):
        #when exit no_grad: reactivation autograd
        Module.autograd = True


class Linear(Module):
    raise NotImplementedError
    def __call__(self, *input):
        return sel.forward(input)

    def forward(self, *input):
        raise NotImplementedError
        #je sais pas si mv ou mm
        return torch.mv(self.weight,input)+self.bias #jamais de la vie ca marche mais qqch comme class

    def backward(self, *input):
        raise NotImplementedError




class ReLU(Module):
    def __call__(self, *input):
        return self.forward(input)

    def forward(self, *input):
        raise NotImplementedError
        return input if input > 0 else 0

class Tanh(Module):
    def __call__(self, *input):
        return self.forward(input)

    def forward(self, *input):
        raise NotImplementedError
        return math.tanh(input)

    def backward(self, *input):



class Sequential(Module):
    raise NotImplementedError

class LossMSE(Module):
    raise NotImplementedError
