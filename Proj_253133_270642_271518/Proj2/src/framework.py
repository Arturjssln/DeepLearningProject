import torch.empty

class Module(object):
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

    def zero_grad(self):
        raise NotImplementedError

    def no_grad(self):
        raise NotImplementedError


class Linear(Module):
    raise NotImplementedError

class ReLU(Module):
    raise NotImplementedError

class Tanh(Module):
    raise NotImplementedError

class Sequential(Module):
    raise NotImplementedError

class LossMSE(Module):
    raise NotImplementedError
