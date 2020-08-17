import torch
import math
import warnings
from collections import OrderedDict
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
    # Training Mode boolean
    training = True

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

    def train(self):
        """
        Set training mode to True
        """
        Module.training = True

    def eval(self):
        """
        Set training mode to False
        """
        Module.training = False

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
        super().__setattr__(name, value)
        # If attribut is a Module, add it to the parameters
        if issubclass(type(value), Module):
            self._parameters[name] = value

    def save(self, path='parameters.pt'):
        """
        Save parameters of a model
        """
        torch.save(self._parameters, path)

    def load(self, path='parameters.pt'):
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
        self._params = {}

    def _init_params(self):
        """
        Initializes the params (Abstract implementation)
        """
        pass

    def reset_grad(self):
        for key in self._params:
            self._params[key].reset_grad()

    def parameters(self):
        params = []
        for _, param in self._params.items():
            params.append(param)
        return params


class Linear(Layer):
    """
    Linear layer
    """
    def __init__(self, dim_in, dim_out):
        super(Linear, self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self._init_params()

    def _init_params(self):
        """
        Initialize weights and bias
        """
        scale = 1 / math.sqrt(self.dim_in)
        self._params['W'] = Parameter(torch.empty(self.dim_in, self.dim_out).uniform_(-scale, scale))
        self._params['b'] = Parameter(torch.empty(1, self.dim_out).uniform_(-scale, scale))

    def forward(self, *input):
        if len(input) > 1:
            warnings.warn(
                "Input for Linear must be composed of only one element, supplementary arguments are ignored.")
        self._store['x']  = input[0]
        out = torch.mm(self._store['x'], self._params['W'].p) + self._params['b'].p
        return out

    def local_grad(self):
        dx_local = self._params['W'].p
        dw_local = self._store['x']
        db_local = torch.ones_like(self._params['b'].p)
        return {'x': dx_local, 'W': dw_local, 'b': db_local}

    def backward(self, *gradwrtoutput):
        dx = torch.mm(gradwrtoutput[0], self._grad['x'].t())
        dw = torch.mm(self._grad['W'].t(), gradwrtoutput[0])
        db = torch.sum(gradwrtoutput[0], dim=0, keepdim=True)
        self._params['W'].grad += dw
        self._params['b'].grad += db
        return dx

    def __repr__(self):
        return "Linear({}, {})".format(self.dim_in, self.dim_out)

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
                "Too few arguments given. Exactly 2 arguments expected, but got {}.".format(len(input)))
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
                "Too few arguments given. Exactly 2 arguments expected, but got {}.".format(len(input)))
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
        super(Sequential, self).__init__()
        self.layers = list(layers)

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


class Conv2d(Layer):
    """
    Conv2d Module
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0):
        super(Conv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding
        self._init_params()

    def _init_params(self):
        """
        Parameter initialization following 'Xavier initialization'
        """
        scale = math.sqrt(2.0 / ((self.in_channels+self.out_channels)*self.kernel_size[0]*self.kernel_size[1]))
        self._params['W'] = Parameter(torch.empty(size=(self.out_channels, self.in_channels, *self.kernel_size)).normal_(std=scale))
        self._params['b'] = Parameter(torch.zeros(self.out_channels, 1))

    def forward(self, *input):
        if len(input) > 1:
            warnings.warn(
                "Input for Conv2d must be composed of only one element, supplementary arguments are ignored.")
        x = input[0]
        if self.padding:
            x = zero_padding(x, width=self.padding, dimensions=(2, 3))
        # initialize input shape and output shape
        N, _, H_IN, W_IN = tuple(x.size())
        KH, KW = self.kernel_size
        C_OUT, H_OUT, W_OUT = (self.out_channels, 1 + (H_IN - KH)//self.stride, 1 + (W_IN - KW)//self.stride)
        out = torch.zeros((N, C_OUT, H_OUT, W_OUT))
        for n in range(N):
            for channel in range(C_OUT):
                for h, w in product(range(H_OUT), range(W_OUT)):
                    h_offset, w_offset = h*self.stride, w*self.stride
                    local_patch = x[n, :, h_offset:h_offset+KH, w_offset:w_offset+KW]
                    out[n, channel, h, w] = torch.sum(self._params['W'].p[channel] * local_patch) + self._params['b'].p[channel]
        
        self._store['x'] = x
        return out

    def backward(self, *gradwrtoutput):
        x = self._store['x']
        dx = torch.zeros_like(x)
        dout = gradwrtoutput[0]
        N, _, H, W = tuple(dx.size())
        KH, KW = self.kernel_size
        for n in range(N):
            for c_out in range(self.out_channels):
                for h, w in product(range(dout.size(2)), range(dout.size(3))):
                    h_offset, w_offset = h * self.stride, w * self.stride
                    dx[n, :, h_offset:h_offset+KH, w_offset:w_offset+KW] += self._params['W'].p[c_out] * dout[n, c_out, h, w]

        dw = torch.zeros_like(self._params['W'].p)
        for c_out in range(self.out_channels):
            for c_in in range(self.in_channels):
                for h, w in product(range(KH), range(KW)):
                    x_local_patch = x[:, c_in, h:H-KH+h+1:self.stride, w:W-KW+w+1:self.stride]
                    dout_local_patch = dout[:, c_out]
                    dw[c_out, c_in, h, w] = torch.sum(x_local_patch*dout_local_patch)

        db = torch.sum(dout, dim=(0, 2, 3)).reshape(self.out_channels, 1)

        self._params['W'].grad += dw
        self._params['b'].grad += db

        if self.padding == 0:
            return dx
        return dx[:, :, self.padding:-self.padding, self.padding:-self.padding]

    def __repr__(self):
        return "Conv2d(in_channels: {}, out_channels: {}, kernel_size: {}, stride: {}, padding: {})".format(
            self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding)


class MaxPool2d(Layer):
    """
    MaxPool2d Module
    """
    def __init__(self, kernel_size=(2, 2)):
        super(MaxPool2d, self).__init__()
        self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size

    def forward(self, *input):
        if len(input) > 1:
            warnings.warn(
                "Input for MaxPool2d must be composed of only one element, supplementary arguments are ignored.")
        x = input[0]
        N, C, H, W = tuple(x.size())
        KH, KW = self.kernel_size
        out = torch.zeros((N, C, H//KH, W//KW))
        for h, w in product(range(0, H//KH), range(0, W//KW)):
            h_offset, w_offset = h*KH, w*KW
            local_patch = x[:, :, h_offset:h_offset+KH, w_offset:w_offset+KW]
            out[:, :, h, w] = torch.max(torch.max(local_patch, dim=3)[0], dim=2)[0]

        self._store['x'] = x
        self._store['out'] = out
        return out
    
    def local_grad(self):
        x = self._store['x']
        out = self._store['out']
        _, _, H, W = tuple(x.size())
        KH, KW = self.kernel_size
        grad = torch.zeros_like(x)
        # Local derivative is 1 if it is a maximum, 0 otherwise
        for h, w in product(range(0, H//KH), range(0, W//KW)):
            h_offset, w_offset = h*KH, w*KW
            for kh, kw in product(range(KH), range(KW)):
                grad[:, :, h_offset+kh, w_offset+kw] = (x[:, :, h_offset+kh, w_offset+kw] >= out[:, :, h, w])
        return {'x': grad}

    def backward(self, *gradwrtoutput):
        dout = gradwrtoutput[0]
        # Expand the gradient
        dout = torch.repeat_interleave(torch.repeat_interleave(dout, self.kernel_size[0], dim=2), self.kernel_size[1], dim=3)
        return self._grad['x'] * dout
    
    def __repr__(self):
        return "MaxPool2d(kernel_size: {})".format(self.kernel_size)


class BatchNorm2d(Layer):
    """
    BatchNorm2d Module
    """
    def __init__(self, num_features, eps=1e-5):
        super(BatchNorm2d, self).__init__()
        self.eps = eps
        self.num_features = num_features
        self._init_params()

    def _init_params(self):
        self._params['gamma'] = Parameter(torch.ones(size=(1, self.num_features, 1, 1)))
        self._params['beta'] = Parameter(torch.zeros(size=(1, self.num_features, 1, 1)))
        # Empirical moments - grad always zero, never updated
        self._params['mu_av'] = Parameter(torch.zeros(size=(1, self.num_features, 1, 1)))
        self._params['var_av'] = Parameter(torch.zeros(size=(1, self.num_features, 1, 1)))

    def forward(self, *input):
        if len(input) > 1:
            warnings.warn(
                "Input for MaxPool2d must be composed of only one element, supplementary arguments are ignored.")
        x = input[0]
        if Module.training:
            mu = torch.mean(x, dim=(2, 3), keepdim=True)
            var = torch.var(x, dim=(2, 3), keepdim=True) + self.eps
            # Calculate empirical moments
            self._params['mu_av'].p = 0.1 * torch.mean(mu, dim=0, keepdim=True) + 0.9 * self._params['mu_av'].p
            self._params['var_av'].p = 0.1 * torch.mean(var, dim=0, keepdim=True) + 0.9 * self._params['var_av'].p
        else:
            mu = self._params['mu_av'].p
            var = self._params['var_av'].p
        xmu = x - mu
        ivar = 1.0/torch.sqrt(var)
        xhat = xmu * ivar
        gammax = self._params['gamma'].p * xhat
        out = gammax + self._params['beta'].p

        self._store['xmu'] = xmu
        self._store['var'] = var
        self._store['ivar'] = ivar
        self._store['xhat'] = xhat
        return out

    def backward(self, *gradwrtoutput):
        dout = gradwrtoutput[0]
        dgamma = torch.sum(self._store['xhat'] * dout, dim=(0, 2, 3), keepdim=True)
        dbeta = torch.sum(dout, dim=(0, 2, 3), keepdim=True)
        self._params['gamma'].grad += dgamma
        self._params['beta'].grad += dbeta

        N = self.num_features
        dx = (1.0 / N) * self._params['gamma'].p * self._store['ivar'] * (N * dout - dbeta - self._store['xmu'] / \
            self._store['var'] * torch.sum(self._store['xmu'] * dout, dim=(0, 2, 3), keepdim=True))
        return dx
    
    def __repr__(self):
        return "BatchNorm2d(num_features: {}, epsilon: {})".format(self.num_features, self.eps)


class Dropout(Layer):
    """
    Dropout Module
    """
    def __init__(self, p=0.5, inplace=False):
        super(Dropout, self).__init__()
        if p < 0 or p > 1:
            raise ValueError("Dropout probability has to be between 0 and 1, but got {}".format(p))
        self.p = p
        self.inplace = inplace

    def forward(self, *input):
        if len(input) > 1:
            warnings.warn(
                "Input for Dropout must be composed of only one element, supplementary arguments are ignored.")
        x = input[0]
        # If mode eval, dropout do nothing
        if not Module.training:
            self._store['drop'] = torch.ones_like(x)
            return x

        drop = (torch.empty(x.size()).uniform_() > self.p).type(torch.IntTensor)
        # Usage of inverted dropout
        if self.inplace:
            x *= drop / (1 - self.p)
            out = x
        else:
            out = x * drop / (1 - self.p)

        self._store['drop'] = drop * (1-self.p)
        return out

    def local_grad(self):
        # Usage of inverted dropout
        return {'x': self._store['drop']}

    def backward(self, *gradwrtoutput):
        dout = gradwrtoutput[0]
        return self._grad['x'] * dout
    
    def __repr__(self):
        return "Dropout(p: {}, inplace: {})".format(self.p, self.inplace)


#### UTILITY FUNCTION ####
def zero_padding(x, width, dimensions):
    width = tuple(0 if i not in dimensions else width for i in range(x.ndim)) if isinstance(width, int) else width
    size = tuple(x.size(i) + 2*w for i, w in enumerate(width))
    x_pad = torch.zeros(size)

    idx = [slice(width[dim], x_pad.size(dim) - width[dim]) for dim in range(x.ndim)]
    x_pad[idx] = x
    return x_pad
