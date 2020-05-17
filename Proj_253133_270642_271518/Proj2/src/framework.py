import torch 
import math
from collections import OrderedDict
import warnings 

class Module(object):
    def __init__(self):
        # initializing cache for intermediate results
        # helps with gradient calculation in some cases
        self._cache = OrderedDict()
        # cache for gradients
        self._grad = OrderedDict()
        self._parameters = OrderedDict()

    def __call__(self, *input):
        # calculating output
        output = self.forward(*input)
        # calculating and caching local gradients
        self._grad = self.local_grad(*input)
        return output
    

    def forward(self, *input):
        """
        Forward pass of the function. Calculates the output value and the
        gradient at the input as well.
        """
        raise NotImplementedError

    def backward(self, *gradwrtoutput):
        raise NotImplementedError

    def local_grad(self, *input):
        pass

    def parameters(self):
        parameters = []
        for key in self._parameters:
            parameters.extend(self._parameters[key].parameters())
        return parameters

    def __repr__(self):
        out = self.__class__.__name__ + ' :\n'
        for key in self._parameters:
            out += self._parameters[key].__repr__() + '\n'
        return out
    
    __str__ = __repr__

    def __setattr__(self, name, value):
        super(Module, self).__setattr__(name, value)
        # If attribut is a Module, add it to the parameters
        if issubclass(type(value), Module):
            self._parameters[name] = value


class ReLU(Module):
    def __init__(self, *input):
        super().__init__()

    def forward(self, *input):
        if len(input) > 1:
            warnings.warn("Input for ReLU must be composed of only one element, supplementary arguments are ignored.")
        input = input[0]
        return input*(input > 0)

    def local_grad(self, *input):
        return {'input': 1*(input[0] > 0)}

    def backward(self, dy):
        return dy * self._grad['input']

    def __repr__(self):
        return "ReLU()"



class Tanh(Module):
    def __init__(self, *input):
        super().__init__()

    def forward(self, *input):
        if len(input) > 1:
            warnings.warn(
                "Input for Tanh must be composed of only one element, supplementary arguments are ignored.")
        input = input[0]
        return math.tanh(input)

    def backward(self, dy):
        return dy * self._grad['input']

    def local_grad(self, *input):
        s = 1 - math.tanh(input)**2
        return {'input': s}

    def __repr__(self):
        return "Tanh()"


class Layer(Module):
    def __init__(self, *input):
        super().__init__()
        self.params = {}

    def _init_params(self, *args):
        """
        Initializes the params.
        """
        pass

    def parameters(self):
        params = []
        for _, param in self.params.items():
            params.append(param)
        return params


class Linear(Layer):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.in_dim = dim_in
        self.out_dim = dim_out
        self._init_params(dim_in, dim_out)

    def _init_params(self, dim_in, dim_out, std=1e-6):
        scale = 1 / math.sqrt(dim_in)
        self.params['W'] = Parameter()
        self.params['b'] = Parameter()
        self.params['W'].p = torch.empty(dim_in, dim_out).uniform_(-std, std)
        self.params['b'].p = torch.empty(1, dim_out).normal_(-std, std)

    def forward(self, *input):
        if len(input) > 1:
            warnings.warn(
                "Input for Linear must be composed of only one element, supplementary arguments are ignored.")
        input = input[0]
        output = torch.mm(input, self.params['W'].p) + self.params['b'].p
        # caching variables for backprop
        self._cache['input'] = input
        self._cache['output'] = output
        return output

    def backward(self, *dy):
        # calculating the global gradient, to be propagated backwards
        dx = torch.mm(*dy, self._grad['input'].t())
        dw = torch.mm(self._grad['W'].t(), *dy)
        db = torch.sum(*dy, dim=0, keepdim=True)
        # caching the global gradients
        self.params['W'].grad = dw
        self.params['b'].grad = db
        return dx

    def local_grad(self, *input):
        dx_local = self.params['W'].p
        dw_local = self._cache['input']
        db_local = torch.ones_like(self.params['b'].p)
        return {'input': dx_local, 'W': dw_local, 'b': db_local}

    def __repr__(self):
        return "Linear({}, {})".format(self.in_dim, self.out_dim)

class Loss(Module):

    def backward(self, *input):
        return self._grad['input']



class MSELoss(Loss):
    def forward(self, *input):
        if len(input) < 2:
            raise RuntimeError(
                "Too few arguments given. Exactly 2 arguments expected.")
        if len(input) > 2:
            warnings.warn(
                "Input for Linear must be composed of exactly two arguments, supplementary arguments are ignored.")
        # calculating MSE loss
        loss =  torch.sum((input[0]-input[1])**2, dim=1, keepdim=True).mean()
        return loss

    def local_grad(self, *input):
        return {'input': 2*(input[0]-input[1])/input[0].shape[0]}

    def __repr__(self):
        return "MSELoss()"

class Parameter(object):
    def __init__(self):
        self.p = None
        self.grad  = None

class Sequential(Module):
    def __init__(self, *layers):
        self.layers = [layer for layer in layers]

    def parameters(self):
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params

    def forward(self, *input):
        if len(input) > 1:
            warnings.warn(
                "Input for Sequential must be composed of only one element, supplementary arguments are ignored.")
        out = input[0]
        for layer in self.layers:
            out = layer(out)
        return out

    def backward(self, dy):
        dout = dy
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout

    def __repr__(self):
        out = "Sequential(\n"
        for i, layer in enumerate(self.layers):
            out += '    ({}) '.format(i) + layer.__repr__() + '\n'
        out += ')'
        return out
