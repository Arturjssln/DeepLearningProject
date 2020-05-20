import torch 
import math
from collections import OrderedDict
import warnings 


class Parameter(object):
    """
    Implementation of a parameter
    """
    def __init__(self, param):
        """
        Parameter is composed of its value and its gradient
        """
        self.p = param
        self.grad = torch.zeros_like(self.p)

    def reset_grad(self):
        self.grad = torch.zeros_like(self.p)


class Module(object):
    """
    Abstract model of a framework module
    """
    def __init__(self):
        # initializing storage of values that can be useful for gradient calculation
        self._store = OrderedDict()
        # initializing gradient storage
        self._grad = OrderedDict()
        # keep track of subclasses of Module
        self._parameters = OrderedDict()

    def __call__(self, *input):
        # calculating output (forward path)
        output = self.forward(*input)
        # calculating and storing local gradients
        self._grad = self.local_grad()
        return output
    

    def forward(self, *input):
        """
        Forward pass (Abstract implementation).
        """
        raise NotImplementedError

    def local_grad(self):
        """
        Compute local gradient (Abstract implementation).

        Returns:
            grad: dictionary of local gradient
        """
        return OrderedDict()

    def backward(self, *gradwrtoutput):
        """
        Backward pass (Abstract implementation).
        """
        raise NotImplementedError


    def parameters(self):
        """
        Paramerters of model

        Returns: 
            parameters : list of Module subclasses parameters
        """
        parameters = []
        for key in self._parameters:
            parameters.extend(self._parameters[key].parameters())
        return parameters

    def zero_grad(self):
        """
        Reset gradient of all parameters
        """
        for key in self._parameters:
            self._parameters[key].reset_grad()

    def reset_grad(self):
        """
        Reset gradient of all parameters of a Module (Abstract implementation)
        """
        pass

    def __repr__(self):
        """
        Representation of the model
        """
        out = self.__class__.__name__ + ' :\n'
        for key in self._parameters:
            out += key + ' : ' + self._parameters[key].__repr__() + '\n'
        return out
    
    __str__ = __repr__

    def __setattr__(self, name, value):
        """
        Store Module subclasses attribute 
        """
        super(Module, self).__setattr__(name, value)
        # If attribut is a Module, add it to the parameters
        if issubclass(type(value), Module):
            self._parameters[name] = value

    def save(self, path = './', name = 'parameters'):
        """
        Save parameters of a model
        """
        torch.save(self._parameters, path + name + '.pt')

    def load(self, path = './parameters.pt'):
        """
        Load parameters of a pretrained model
        """
        parameters = torch.load(path)
        for key in parameters:
            self.__setattr__(key, parameters[key])



class ReLU(Module):
    """
    ReLU activation fonction
    """
    def __init__(self):
        super(ReLU, self).__init__()

    def forward(self, *input):
        if len(input) > 1:
            warnings.warn("Input for ReLU must be composed of only one element, supplementary arguments are ignored.")
        self._store['x'] = input[0]
        return self._store['x']*(self._store['x'] > 0)

    def local_grad(self):
        grad = 1*(self._store['x'] > 0)
        return {'x': grad}

    def backward(self, *gradwrtoutput):
        return gradwrtoutput[0] * self._grad['x']

    def __repr__(self):
        return "ReLU()"



class Tanh(Module):
    """
    Tanh activation fonction
    """
    def __init__(self):
        super(Tanh, self).__init__()

    def forward(self, *input):
        if len(input) > 1:
            warnings.warn(
                "Input for Tanh must be composed of only one element, supplementary arguments are ignored.")
        self._store['x'] = input[0]
        return math.tanh(self._store['x'])

    def local_grad(self):
        grad = 1 - math.tanh(self._store['x'])**2
        return {'x': grad}

    def backward(self, *gradwrtoutput):
        return gradwrtoutput[0] * self._grad['x']


    def __repr__(self):
        return "Tanh()"


class Layer(Module):
    """
    Abstract model of a layer
    """
    def __init__(self):
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
    """
    Linear layer
    """
    def __init__(self, dim_in, dim_out):
        super(Linear, self).__init__()
        self.in_dim = dim_in
        self.out_dim = dim_out
        self._init_params(dim_in, dim_out)

    def _init_params(self, dim_in, dim_out):
        """
        Initialize weights and bias
        """
        scale = 1 / math.sqrt(dim_in)
        self.params['W'] = Parameter(torch.empty(dim_in, dim_out).uniform_(-scale, scale))
        self.params['b'] = Parameter(torch.empty(1, dim_out).uniform_(-scale, scale))

    def forward(self, *input):
        if len(input) > 1:
            warnings.warn(
                "Input for Linear must be composed of only one element, supplementary arguments are ignored.")
        self._store['x']  = input[0]
        output = torch.mm(self._store['x'], self.params['W'].p) + self.params['b'].p
        self._store['output'] = output
        return output

    def local_grad(self):
        dx_local = self.params['W'].p
        dw_local = self._store['x']
        db_local = torch.ones_like(self.params['b'].p)
        return {'x': dx_local, 'W': dw_local, 'b': db_local}

    def backward(self, *gradwrtoutput):
        dx = torch.mm(gradwrtoutput[0], self._grad['x'].t())
        dw = torch.mm(self._grad['W'].t(), gradwrtoutput[0])
        db = torch.sum(gradwrtoutput[0], dim=0, keepdim=True)
        self.params['W'].grad += dw
        self.params['b'].grad += db
        return dx

    def __repr__(self):
        return "Linear({}, {})".format(self.in_dim, self.out_dim)

class Loss(Module):
    """
    Abstract model of a loss
    """
    def __init__(self):
        super(Loss, self).__init__()

    def backward(self, *gradwrtoutput):
        return self._grad['x']

class MSELoss(Loss):
    """
    MSE Loss
    """
    def forward(self, *input):
        if len(input) < 2:
            raise RuntimeError(
                "Too few arguments given. Exactly 2 arguments expected.")
        if len(input) > 2:
            warnings.warn(
                "Input for MSELoss must be composed of exactly two arguments, supplementary arguments are ignored.")
        self._store['pred'] = input[0]
        self._store['target'] = input[1]
        loss = torch.sum((self._store['pred']-self._store['target'])** 2, dim=1, keepdim=True).mean()
        return loss

    def local_grad(self):
        size = self._store['pred'].shape[0]
        grad = 2*(self._store['pred']-self._store['target'])/size
        return {'x': grad}

    def __repr__(self):
        return "MSELoss()"


class CrossEntropyLoss(Loss):
    """
    Cross Entropy Loss
    """
    def forward(self, *input):
        if len(input) < 2:
            raise RuntimeError(
                "Too few arguments given. Exactly 2 arguments expected.")
        if len(input) > 2:
            warnings.warn(
                "Input for CrossEntropyLoss must be composed of exactly two arguments, supplementary arguments are ignored.")
        input_ = input[0]
        target_ = input[1]
        # calculate softmax
        exps = torch.exp(input_)
        proba = exps / torch.sum(exps, dim=1, keepdim=True)
        # calculate log likelihood
        log_likelihood = -torch.log(proba[range(target_.size(0)), target_])
        # calculate cross entropy
        crossentropy_loss = torch.mean(log_likelihood)
        # storing for gradient
        self._store['p'] = proba
        self._store['t'] = target_
        return crossentropy_loss

    def local_grad(self):
        grad = self._store['p']
        t = self._store['t']
        grad[range(t.size(0)), t] -= 1
        grad /= t.size(0)
        return {'x': grad}

    def __repr__(self):
        return "CrossEntropyLoss()"


class Sequential(Module):
    """
    Sequential module
    """
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

    def backward(self, *gradwrtoutput):
        dout = gradwrtoutput[0]
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout

    def __repr__(self):
        out = "Sequential(\n"
        for i, layer in enumerate(self.layers):
            out += '    ({}) '.format(i) + layer.__repr__() + '\n'
        out += ')'
        return out
