from torch import empty
import math

class Module(object):
    autograd = True
    def __init__(self):
        # initializing cache for intermediate results
        # helps with gradient calculation in some cases
        self.cache = {}
        # cache for gradients
        self.grad = {}

    def __call__(self, *input):
        # calculating output
        output = self.forward(*input)
        # calculating and caching local gradients
        self.grad = self.local_grad(*input)
        return output


    def forward(self, *input):
        """
        Forward pass of the function. Calculates the output value and the
        gradient at the input as well.
        """
        pass

    def backward(self, *gradwrtoutput):
        pass

    def local_grad(self, *input):
        pass

    def parameters(self):
        return {}

    def criterion(self):
        raise NotImplementedError


class ReLU(Module):
    def forward(self, *input):
        return input if input > 0 else 0

    def local_grad(self, *input):
        return {'input': 1} if input > 0 else {'input': 0}

    def backward(self, dy):
        return dy * self.grad['input']


class Tanh(Module):
    def forward(self, *input):
        return math.tanh(input)

    def backward(self, dy):
        return dy * self.grad['input']

    def local_grad(self, *input):
        s = 1 - math.tanh(input)**2
        return {'input': s}


class Layer(Module):
    def __init__(self, *input):
        super().__init__(*input)
        self.params = {}

    def _init_params(self, *args):
        """
        Initializes the params.
        """
        pass

    def parameters(self):
        return self.params


class Linear(Layer):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self._init_params(in_dim, out_dim)

    def _init_params(self, in_dim, out_dim, std = 1e-6):
        #TODO: IMPROVE THAT
        scale = 1 / math.sqrt(in_dim)
        self.params['W'] = Parameter()
        self.params['b'] = Parameter()
        self.params['W'].p = scale * \
            empty(in_dim, out_dim).normal_(mean=0, std=std)
        self.params['b'].p = scale * \
            empty(in_dim, out_dim).normal_(mean=0, std=std)

    def forward(self, input):
        output = torch.mm(input, self.params['W'].p) + self.params['b'].p
        # caching variables for backprop
        self.cache['input'] = input
        self.cache['output'] = output

        return output

    def backward(self, dy):
        # calculating the global gradient, to be propagated backwards
        dx = torch.mm(dy, self.grad['input'].transpose())
        # calculating the global gradient wrt to paramss
        input = self.cache['input']
        dw = torch.mm(self.grad['W'].p.transpose(), dy)
        db = torch.sum(dy, dim=0, keepdim=True)
        # caching the global gradients
        self.grad['W'].grad = dw
        self.grad['b'].grad = db
        return dx


class Loss(Module):
    def forward(self, input, y):
        pass

    def backward(self):
        return self.grad['input']

    def local_grad(self, input, y):
        pass


class MSELoss(Loss):
    def forward(self, input, y):
         # calculating MSE loss
        loss = ((input-y)**2).mean()

        # caching for backprop
        self.cache['y'] = y

        return loss

    def local_grad(self, input, y):
        return {'input': 2*(input-y).mean()}

class Parameter(object):
    def __init__(self):
        self.p = None
        self.grad  = None
        

class zero_grad(Module):
    raise NotImplementedError

class no_grad(Module):
    def __enter__(self):
        Module.autograd = False

    def __exit__(self, type, value, traceback):
        Module.autograd = True


class Sequential(Module):
    def __init__(self, *layers):
        self.layers = [layer for layer in layers]

    def parameters(self):
        params = []
        for layer in self.layers:
            param = layer.parameters()
            for key in param:
                params.append(param[key])
        return params

    def forward(self, input):
        out = input
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def backward(self, dy):
        dout = dy
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout
