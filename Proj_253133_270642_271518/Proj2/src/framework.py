from torch import empty
import math

class Module(object):
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

    def __repr__(self):
        pass
    
    def __str__(self):
        return self.__repr__()


class ReLU(Module):
    def forward(self, *input):
        return input if input > 0 else 0

    def local_grad(self, *input):
        return {'input': 1} if input > 0 else {'input': 0}

    def backward(self, dy):
        return dy * self.grad['input']

    def __repr__(self):
        return "ReLU()"



class Tanh(Module):
    def forward(self, *input):
        return math.tanh(input)

    def backward(self, dy):
        return dy * self.grad['input']

    def local_grad(self, *input):
        s = 1 - math.tanh(input)**2
        return {'input': s}

    def __repr__(self):
        return "Tanh()"


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
        params = []
        for key, param in self.params.items():
                params.append(param)
        return self.params


class Linear(Layer):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self._init_params(in_dim, out_dim)

    def _init_params(self, in_dim, out_dim, std = 1e-6):
        scale = 1 / math.sqrt(in_dim)
        self.params['W'] = Parameter()
        self.params['b'] = Parameter()
        self.params['W'].p = scale * \
            empty(in_dim, out_dim).normal_(mean=0, std=std)
        self.params['b'].p = scale * \
            empty(1, out_dim).normal_(mean=0, std=std)

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

    def __repr__(self):
        return "Linear({}, {})".format(self.in_dim, self.out_dim)

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
            params += list(layer.parameters())
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

    def __repr__(self):
        out = "Sequential(\n"
        for i, layer in enumerate(self.layers):
            out += '({}) '.format(i) + layer.__repr__() + '\n'
        out += ')'
        return out
