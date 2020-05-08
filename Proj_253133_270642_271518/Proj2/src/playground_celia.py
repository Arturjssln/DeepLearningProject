from torch import empty
import math

class Module(object):
    autograd = True #why?
    def __init__(self, *input):
        # initializing cache for intermediate results
        # helps with gradient calculation in some cases
        self.cache = {}
        # cache for gradients
        self.grad = {}

    def __call__(self, *input):
        # calculating output
        output = self.forward(*args, **kwargs)
        # calculating and caching local gradients
        self.grad = self.local_grad(*args, **kwargs)
        return output

    def forward(self, *input):
        """
        Forward pass of the function. Calculates the output value and the
        gradient at the input as well.
        """
        pass


    def backward(self, *gradwrtoutput):
        pass

    def param(self):
        return []

    def criterion(self):
        raise NotImplementedError

    def local_grad(self, *args, **kwargs):
        """
        Calculates the local gradients of the function at the given input.
        Returns:
            grad: dictionary of local gradients.
        """
        pass 



class zero_grad(Module):
    raise NotImplementedError



class no_grad(Module):
    def __enter__(self):
        #when no_grad called: autograd desactivated
        Module.autograd = False

    def __exit__(self, type, value, traceback):
        #when exit no_grad: reactivation autograd
        Module.autograd = True


class ReLU(Module):
    def forward(self, *input):
        return input if input > 0 else 0

    def local_grad(self, *input):
        return {'input': 1} if input > 0 else {'input': 0}

    def backward(self, dy):
        return dy * self.grad['input']     


class Tanh(Module):
    def forward(self, *input):
        raise NotImplementedError
        return math.tanh(input)

    def backward(self, dy):
        return dy * self.grad
    
    def local_grad(self, input):
        s=1-math.tanh(input)**2
        grads={'input':s}
        return grads



class Layer(Module):
    """
    Abstract model of a neural network layer. In addition to Function, a Layer
    also has weights and gradients with respect to the weights.
    """
    def __init__(self, *input):
        super().__init__(*input)
        self.weight = {} # weights + bias
        self.weight_update = {}

    def _init_weights(self, *input):
        """
        Initializes the weights.
        """
        pass

    def _update_weights(self, lr):
        """
        Updates the weights using the corresponding _global_ gradients computed during
        backpropagation.
        Args:
             lr: float. Learning rate.
        """
        for weight_key, weight in self.weight.items():
            self.weight[weight_key] = self.weight[weight_key] - lr * self.weight_update[weight_key]


class Linear(Layer):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self._init_weights(in_dim, out_dim)

    #TODO: improve 
    def _init_weights(self, in_dim, out_dim):
        scale = 1 / sqrt(in_dim)
        epsilon=1e-6
        self.weight['W'] = scale * torch.empty(in_dim, out_dim).normal_(mean=0,std=epsilon)
        self.weight['b'] = scale * torch.empty(1, out_dim).normal_(mean=0,std=epsilon)

    def forward(self, input):
        """
        Forward pass for the Linear layer.
        Args:
            X: numpy.ndarray of shape (n_batch, in_dim) containing
                the input value.
        Returns:
            Y: numpy.ndarray of shape of shape (n_batch, out_dim) containing
                the output value.
        """

        output = torch.mm(input, self.weight['W']) + self.weight['b']

        # caching variables for backprop
        self.cache['input'] = input
        self.cache['output'] = output

        return output

    def backward(self, dy):
        """
        Backward pass for the Linear layer.
        Args:
            dY: numpy.ndarray of shape (n_batch, n_out). Global gradient
                backpropagated from the next layer.
        Returns:
            dX: numpy.ndarray of shape (n_batch, n_out). Global gradient
                of the Linear layer.
        """
        # calculating the global gradient, to be propagated backwards
        dx = torch.mm(dy, self.grad['input'].transpose())
        # calculating the global gradient wrt to weights
        input = self.cache['input']
        dw = torch.mm(self.grad['W'].transpose(), dy)
        db = torch.sum(dy, dim=0, keepdim=True)
        # caching the global gradients
        self.weight_update = {'W': dw, 'b': db}

        return dx
        

class Loss(Module):
    def forward(self, input, y):
        """
        Computes the loss of x with respect to y.
        Args:
            X: numpy.ndarray of shape (n_batch, n_dim).
            Y: numpy.ndarray of shape (n_batch, n_dim).
        Returns:
            loss: numpy.float.
        """
        pass

    def backward(self):
        """
        Backward pass for the loss function. Since it should be the final layer
        of an architecture, no input is needed for the backward pass.
        Returns:
            gradX: numpy.ndarray of shape (n_batch, n_dim). Local gradient of the loss.
        """
        return self.grad['input']

    def local_grad(self, input, y):
        """
        Local gradient with respect to X at (X, Y).
        Args:
            X: numpy.ndarray of shape (n_batch, n_dim).
            Y: numpy.ndarray of shape (n_batch, n_dim).
        Returns:
            gradX: numpy.ndarray of shape (n_batch, n_dim).
        """
        pass


class MSELoss(Loss):
    def forward(self, input, y):
        """
        Computes the cross entropy loss of x with respect to y.
        Args:
            X: numpy.ndarray of shape (n_batch, n_dim).
            y: numpy.ndarray of shape (n_batch, 1). Should contain class labels
                for each data point in x.
        Returns:
            crossentropy_loss: numpy.float. Cross entropy loss of x with respect to y.
        """
        # calculating MSEloss
        MSE_loss = ((input- y)**2).mean()

        # caching for backprop
        self.cache['y'] = y

        return MSE_loss

    def local_grad(self, input, y):
        grads = {'input': 2 * (input - y).mean()}
        return grads



class Sequential(Module):
    raise NotImplementedError





