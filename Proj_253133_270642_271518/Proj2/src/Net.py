import torch
import framework as ff

class Net(ff.Module):
    def __init__(self, nb_nodes, act_fct = ff.ReLU()):
        super(Net, self).__init__()
        self.best_epoch = 0
        self.sumloss = []
        self.train_error = []
        self.test_error = []

        self.linear_layers = ff.Sequential(ff.Linear(2, nb_nodes), act_fct, \
                                            ff.Linear(nb_nodes, nb_nodes), act_fct, \
                                            ff.Linear(nb_nodes, nb_nodes), act_fct, \
                                            ff.Linear(nb_nodes, 2))
    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        return self.linear_layers(x)

    def backward(self, criterion):
        d = criterion.backward()
        d = self.linear_layers.backward(d)
        return d

    def train(  self, \
                train_input, train_target, \
                test_input = None, test_target = None, \
                batch_size = 10, epoch = 50, \
                eta = 1e-2, criterion = ff.MSELoss(), print_skip = 5):
        """
        Training method
        """
        for e in range(epoch):
            sum_loss = 0
            # We do this with mini-batches
            for b in range(0, train_input.size(0), batch_size):
                # Forward + save local grad for each layer
                output = self(train_input.narrow(0, b, batch_size))
                # Forward + save local grad of loss layer
                loss = criterion(output, train_target.narrow(0, b, batch_size))
                sum_loss = sum_loss + loss.item()
                self.backward(criterion) 
                for p in self.parameters():
                    p.p -= eta * p.grad


            self.sumloss.append(sum_loss)
            self.train_error.append(self.compute_error_rate(train_input, train_target, batch_size))
            self.test_error.append(self.compute_error_rate(test_input, test_target, batch_size))

            if e%print_skip == 0:
                print("Epoch #{:2d} --> Total train loss : {:.03f} ".format(e,  self.sumloss[-1]))
                print("------------> Train error rate : {:.02f}% ".format(self.train_error[-1]*100))
                if test_input is not None and test_target is not None:
                    print("------------> Test error rate : {:.02f}% ".format(self.test_error[-1]*100))
                print("-------------------------------------------------")

            #Save best epoch
            if self.test_error[self.best_epoch] > self.test_error[-1]:
                self.best_epoch = e

        print("** BEST SCORE --> Epoch #{:2d}: \n*  train_error: {:.02f}%, \n*  test_error: {:.02f}%"\
            .format(self.best_epoch, self.train_error[self.best_epoch]*100, self.test_error[self.best_epoch]*100))

    def compute_error_rate(self, input, target, batch_size):
        '''
        Computing error rate givin an input and its target
        '''
        error = 0.0
        for b in range(0, target.size(0), batch_size):
            prediction = self(input.narrow(0, b, batch_size))
            _, predicted_classes = prediction.max(dim = 1)
            # Calculate test error
            for pred, t in zip(predicted_classes, target.narrow(0, b, batch_size)):
                if pred.item() != t.argmax():
                    error += 1
        error /= target.size(0)
        return error
