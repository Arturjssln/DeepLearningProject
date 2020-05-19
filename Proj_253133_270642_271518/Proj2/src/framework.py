import torch 
import math
from collections import OrderedDict
import warnings 

class Module(object):
    def __init__(self):
        # initializing cache for intermediate results
        # helps with gradient calculation in some cases
        self._store = OrderedDict()
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
        return OrderedDict()

    def parameters(self):
        parameters = []
        for key in self._parameters:
            parameters.extend(self._parameters[key].parameters())
        return parameters

    def zero_grad(self):
        for key in self._parameters:
            self._parameters[key].reset_grad()

    def reset_grad(self):
        pass

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
        super(ReLU, self).__init__()

    def forward(self, *input):
        if len(input) > 1:
            warnings.warn("Input for ReLU must be composed of only one element, supplementary arguments are ignored.")
        x = input[0]
        return x*(x > 0)

    def local_grad(self, *input):
        return {'x': 1*(input[0] > 0)}

    def backward(self, dy):
        return dy * self._grad['x']

    def __repr__(self):
        return "ReLU()"



class Tanh(Module):
    def __init__(self, *input):
        super(Tanh, self).__init__()

    def forward(self, *input):
        if len(input) > 1:
            warnings.warn(
                "Input for Tanh must be composed of only one element, supplementary arguments are ignored.")
        x = input[0]
        return math.tanh(x)

    def backward(self, dy):
        return dy * self._grad['x']

    def local_grad(self, *input):
        s = 1 - math.tanh(input)**2
        return {'x': s}

    def __repr__(self):
        return "Tanh()"


class Layer(Module):
    def __init__(self, *input):
        super(Layer, self).__init__()
        self.params = {}

    def _init_params(self, *args):
        """
        Initializes the params.
        """
        pass

    def reset_grad(self):
        for key in self.params:
            self.params[key].reset_grad()

    def parameters(self):
        params = []
        for _, param in self.params.items():
            params.append(param)
        return params


class Linear(Layer):
    def __init__(self, dim_in, dim_out):
        super(Linear, self).__init__()
        self.in_dim = dim_in
        self.out_dim = dim_out
        self._init_params(dim_in, dim_out)

    def _init_params(self, dim_in, dim_out):
        scale = 1 / math.sqrt(dim_in)
        self.params['W'] = Parameter(torch.empty(dim_in, dim_out).uniform_(-scale, scale))
        self.params['b'] = Parameter(torch.empty(1, dim_out).uniform_(-scale, scale))

    def forward(self, *input):
        if len(input) > 1:
            warnings.warn(
                "Input for Linear must be composed of only one element, supplementary arguments are ignored.")
        x = input[0]
        output = torch.mm(x, self.params['W'].p) + self.params['b'].p
        # Storing for backprop
        self._store['x'] = x
        self._store['output'] = output
        return output

    def backward(self, *dy):
        dx = torch.mm(*dy, self._grad['x'].t())
        dw = torch.mm(self._grad['W'].t(), *dy)
        db = torch.sum(*dy, dim=0, keepdim=True)
        self.params['W'].grad += dw
        self.params['b'].grad += db
        return dx

    def local_grad(self, *input):
        dx_local = self.params['W'].p
        dw_local = self._store['x']
        db_local = torch.ones_like(self.params['b'].p)
        return {'x': dx_local, 'W': dw_local, 'b': db_local}

    def __repr__(self):
        return "Linear({}, {})".format(self.in_dim, self.out_dim)

class Loss(Module):
    def __init__(self):
        super(Loss, self).__init__()

    def backward(self, *input):
        return self._grad['x']

class MSELoss(Loss):
    def forward(self, *input):
        if len(input) < 2:
            raise RuntimeError(
                "Too few arguments given. Exactly 2 arguments expected.")
        if len(input) > 2:
            warnings.warn(
                "Input for MSELoss must be composed of exactly two arguments, supplementary arguments are ignored.")
        loss =  torch.sum((input[0]-input[1])**2, dim=1, keepdim=True).mean()
        return loss

    def local_grad(self, *input):
        return {'x': 2*(input[0]-input[1])/input[0].shape[0]}

    def __repr__(self):
        return "MSELoss()"


class CrossEntropyLoss(Loss):
    def forward(self, *input):
        if len(input) < 2:
            raise RuntimeError(
                "Too few arguments given. Exactly 2 arguments expected.")
        if len(input) > 2:
            warnings.warn(
                "Input for CrossEntropyLoss must be composed of exactly two arguments, supplementary arguments are ignored.")
        input_ = input[0]
        target_ = input[1]
        exps = torch.exp(input_)
        proba = exps / torch.sum(exps, dim=1, keepdim=True)
        log_likelihood = -torch.log(proba[range(target_.size(0)), target_])
        crossentropy_loss = torch.mean(log_likelihood)

        # Storing for backprop
        self._store['p'] = proba
        self._store['t'] = target_

        return crossentropy_loss

    def local_grad(self, *input):
        grad = self._store['p']
        t = self._store['t']
        grad[range(t.size(0)), t] -= 1
        grad /= t.size(0)
        return {'x': grad}

    def __repr__(self):
        return "CrossEntropyLoss()"


class Parameter(object):
    def __init__(self, param):
        self.p = param
        self.grad = torch.zeros_like(self.p)

    def reset_grad(self):
        self.grad = torch.zeros_like(self.p)

class Sequential(Module):
    def __init__(self, *layers):
        self.layers = [layer for layer in layers]

    def parameters(self):
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params
    
    def reset_grad(self):
        for layer in self.layers:
            layer.reset_grad()

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
