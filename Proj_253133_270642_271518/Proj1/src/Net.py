import torch
from torch import nn
from torch.nn import functional as F

class Net(nn.Module):
    def __init__(self, architecture, skip_connections, batch_normalization):
        super(Net, self).__init__()
        #TODO: Implement structure

        self.test_error = []
        self.train_error = []
        self.sumloss = []
        self.best_epoch = 0

    def forward(self, x):
        #TODO: Implement forward path
        return x

    def train(  self, \
                train_input, train_target, test_input = None, test_target = None, \
                batch_size = 10, epoch = 25, \
                eta = 1e-1, criterion = nn.MSELoss(), print_skip = 5):

        optimizer = torch.optim.SGD(self.parameters(), lr = eta)
        for e in range(epoch):
            sum_loss = 0
            # We do this with mini-batches
            for b in range(0, train_input.size(0), batch_size):
                output = model(train_input.narrow(0, b, batch_size))
                loss = criterion(output, train_target.narrow(0, b, batch_size))
                sum_loss = sum_loss + loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            self.sumloss.append(sum_loss)

            self.train_error.append(self.compute_error_rate(train_input, train_target, batch_size))
            self.test_error.append(self.compute_error_rate(test_input, test_target, batch_size))

            if e%print_skip == 0:
                print("Epoch #{:d} --> Total train loss : {:.03f} ".format(e,  self.sumloss[-1]))
                print("------------> Train error rate : {:.02f}% ".format(self.train_error[-1]*100))
                if test_input is not None and test_target is not None:
                    print("------------> Test error rate : {:.02f}% ".format(self.test_error[-1]*100))

            #Save best epoch
            if self.test_error[self.best_epoch] > self.test_error[-1]:
                self.best_epoch = e;

        print("BEST SCORE --> Epoch #{:d}: train_error: {:.02f}%, test_error: {:.02f}%".format(self.best_epoch, self.train_error[self.best_epoch]*100, self.test_error[self.best_epoch]*100))

    def compute_error_rate(self, input, target, batch_size):
        error = 0.0
        sample_size = 0
        predicition = self.forward(input[::batch_size, :, :, :])
        # Calculate test error
        for x, t in zip(predicition, target[::batch_size, :]):
            sample_size += 1
            if torch.argmax(x) != torch.argmax(t):
                error += 1.0
        error /= sample_size
        self.error.append(error)
