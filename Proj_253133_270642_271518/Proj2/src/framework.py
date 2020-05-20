import torch
import math
from collections import OrderedDict
import warnings
from itertools import product


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
        raise NotImplementedError

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
        out = torch.mm(self._store['x'], self.params['W'].p) + self.params['b'].p
        return out

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


class Conv2D(Layer):
    def __init__(self, channel_in, channel_out, kernel_size=3, stride=1, padding=0):
        super(Conv2D, self).__init__()
        self.channel_in = channel_in
        self.channel_out = channel_out
        self.stride = stride
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.padding = padding
        self._init_params(channel_in, channel_out, self.kernel_size)

    def _init_params(self, channel_in, channel_out, kernel_size):
        scale = 2/math.sqrt(channel_in*kernel_size[0]*kernel_size[1])
        self.params['W'] = Parameter(torch.empty(size=(channel_out, channel_in, *kernel_size)).normal_(std=scale))
        self.params['b'] = Parameter(torch.zeros(channel_out, 1))

    def forward(self, *input):
        if len(input) > 1:
            warnings.warn(
                "Input for Conv2D must be composed of only one element, supplementary arguments are ignored.")
        self._store['x'] = input[0]
        if self.padding:
            self._store['x'] = zero_padding(self._store['x'], width=self.padding, dimensions=(2, 3))

        # initialize input shape and output shape
        N, C_in, H_in, W_in = tuple(self._store['x'].size())
        KH, KW = self.kernel_size
        C_out, H_out, W_out = (self.channel_out, 1 + (H - KH)//self.stride, 1 + (W - KW)//self.stride)
        out = torch.zeros((N, C_out, H_out, W_out))
        for n in range(N):
            for channel in range(C_out):
                for h, w in product(range(H_out), range(W_out)):
                    h_offset, w_offset = h*self.stride, w*self.stride
                    local_patch = self._store['x'][n, :, h_offset:h_offset+KH, w_offset:w_offset+KW]
                    out[n, channel, h, w] = torch.sum(self.params['W'][channel] * local_patch) + self.params['b'][channel]
        return out

    def backward(self, *gradwrtoutput):
        x = self._store['x']
        dx = torch.zeros_like(x)
        dout = gradwrtoutput[0]
        N, C, H, W = dx.shape
        KH, KW = self.kernel_size
        for n in range(N):
            for c_out in range(self.channel_out):
                for h, w in product(range(dout.size(2)), range(dout.size(3))):
                    h_offset, w_offset = h * self.stride, w * self.stride
                    dx[n, :, h_offset:h_offset+KH, w_offset:w_offset+KW] += self.params['W'][c_out] * dout[n, c_out, h, w]

        dw = torch.zeros_like(self.params['W'])
        for c_out in range(self.channel_out):
            for c_in in range(self.channel_in):
                for h, w in product(range(KH), range(KW)):
                    x_local_patch = x[:, c_in, h:H-KH+h+1:self.stride, w:W-KW+w+1:self.stride]
                    dout_local_patch = dout[:, c_out]
                    dw[c_out, c_in, h, w] = torch.sum(x_local_patch*dout_local_patch)

        db = torch.sum(dout, dim=(0, 2, 3)).reshape(self.channel_out, 1)

        self.params['W'].grad += dw
        self.params['b'].grad += db
        return dx[:, :, self.padding:-self.padding, self.padding:-self.padding]


class MaxPool2D(Module):
    def __init__(self, kernel_size=(2, 2)):
        super(MaxPool2D, self).__init__()
        self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size

    def forward(self, x):
        N, C, H, W = tuple(x.size())
        KH, KW = self.kernel_size

        grad = torch.zeros_like(x)
        out = torch.zeros((N, C, H//KH, W//KW))

        # for n in range(N):
        for h, w in product(range(0, H//KH), range(0, W//KW)):
            h_offset, w_offset = h*KH, w*KW
            local_patch = x[:, :, h_offset:h_offset+KH, w_offset:w_offset+KW]
            out[:, :, h, w] = torch.max(local_patch, axis=(2, 3))

        self._store['x'] = x
        self._store['out'] = out
        return out
    
    def local_grad(self):
        N, C, H, W = tuple(x.size())
        KH, KW = self.kernel_size
        x = self._store['x']
        out = self._store['out']
        grad = torch.zeros_like(x)
        for h, w in product(range(0, H//KH), range(0, W//KW)):
            h_offset, w_offset = h*KH, w*KW
            for kh, kw in product(range(KH), range(KW)):
                grad[:, :, h_offset+kh, w_offset+kw] = (x[:, :, h_offset+kh, w_offset+kw] >= out[:, :, h, w])
        return {'x': grad}

    def backward(self, *gradwrtoutput):
        dout = gradwrtoutput[0]
        dout = torch.repeat_interleave(torch.repeat_interleave(dout, repeats=self.kernel_size[0], dim=2), repeats=self.kernel_size[1], dim=3)
        return self._grad['x'] * dout



#### UTILITY FUNCTION ####
def zero_padding(x, width, dimensions):
    width = (0 if i not in dimensions else width for i in range(x.ndim)) if isinstance(width, int) else width
    size = (x.size(i) + 2*w for i, w in enumerate(width))
    x_pad = torch.zeros(size)

    idx = [slice(width[dim], x_pad.size(dim) - width[dim])
           for dim in range(x.ndim)]
    x_pad[idx] = x
    return x_pad
