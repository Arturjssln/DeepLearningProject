import torch
from torch import nn
from torch.nn import functional as F

class Net(nn.Module):
    def __init__(self, architecture, skip_connections, batch_normalization):
        super(Net, self).__init__()

        #TODO: Implement structure
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3)
        self.fc1 = nn.Linear(16, 200)
        self.fc2 = nn.Linear(200, 10)

        self.test_error = []
        self.train_error = []
        self.sumloss = []
        self.best_epoch = 0

    def forward(self, x):
        #TODO: Implement forward path
        x = F.relu(F.max_pool2d(self.conv1(x), kernel_size=3))
        x = F.relu(F.max_pool2d(self.conv2(x), kernel_size=2))
        x = F.relu(self.fc1(x.view(-1, 16)))
        x = self.fc2(x)
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
                output = self(train_input.narrow(0, b, batch_size))
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
        for b in range(0, input.size(0), batch_size):
            predicition = self(input.narrow(0, b, batch_size))
            _, predicted_classes = predicition.max(1)
            # Calculate test error
            for pred, t in zip(predicted_classes, target.narrow(0, b, batch_size)):
                if pred != t.item():
                    error += 1
        error /= input.size(0)
        return error
