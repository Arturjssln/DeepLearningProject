from torch import empty


class Module(object):
    autograd = True
    def __init__(self):
        self.var = 1

    @classmethod
    def print_grad(cls):
        print("autograd =", Module.autograd)



class no_grad(Module):
    def __enter__(self):
        Module.autograd = False

    def __exit__(self ,type, value, traceback):
        Module.autograd = True


class Cercle(Module):
    def __init__(self):
        super(Cercle, self).__init__

print(torch.tanh(5))

c = Cercle()
c.print_grad()
with no_grad():
     c.print_grad()
c.print_grad()
