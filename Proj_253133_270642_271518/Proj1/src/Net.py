import torch
from torch import nn
from torch.nn import functional as F

class Net(nn.Module):
    def __init__(self, architecture, *param):
        '''
        Initialization
        param contains :
        (usefull parameters will have values, other will be None)
        nb_classes, nb_residual_blocks, nb_channels,
        kernel_size, skip_connections, batch_normalization,
        nb_linear_layers, nb_nodes, optimizer
        '''

        super(Net, self).__init__()
        self.test_error = []
        self.train_error = []
        self.sumloss = []
        self.best_epoch = 0
        self.architecture = architecture

        # Usefull parameters will have values, other will be None
        nb_classes, nb_residual_blocks, \
        nb_channels, kernel_size, \
        skip_connections, batch_normalization, \
        nb_linear_layers, nb_nodes, optimizer = param

        self.optimizer = optimizer

        # default architecture
        if architecture is None:
            #TODO: Implement default structure
            self.conv1 = nn.Conv2d(1, 8, kernel_size=3)
            self.conv2 = nn.Conv2d(8, 16, kernel_size=3)
            self.fc1 = nn.Linear(16, 200)
            self.fc2 = nn.Linear(200, nb_classes)

        # Linear fully connected architecture
        elif architecture == 'linear':
            modules = []
            modules.append(nn.Linear(14**2, nb_nodes))
            modules.append(nn.ReLU())
            for _ in range(nb_linear_layers - 1):
                modules.append(nn.Linear(nb_nodes, nb_nodes))
                modules.append(nn.ReLU())
            modules.append(nn.Linear(nb_nodes, 10))
            self.layers = nn.Sequential(*modules)

        # ResNet architecture
        elif architecture == 'resnet':
            self.batch_normalization = batch_normalization
            self.conv = nn.Conv2d(1, nb_channels, kernel_size = kernel_size, padding = (kernel_size - 1) // 2)
            if batch_normalization:
                self.bn = nn.BatchNorm2d(nb_channels)
            self.resnet_blocks = nn.Sequential(
                *(ResNetBlock(nb_channels, kernel_size, skip_connections, batch_normalization) for _ in range(nb_residual_blocks))
            )
            self.fc = nn.Linear(nb_channels, nb_classes)

        elif architecture == 'lenet' or architecture == 'alexnet':
            raise NotImplementedError

        elif architecture == 'inception':
            raise NotImplementedError

        elif architecture == 'inceptionresnet':
            raise NotImplementedError

        elif architecture == 'xception':
            raise NotImplementedError

        else:
            raise NameError('Unknown architecture')

    def forward(self, x):
        '''
        Forward path
        '''
        # default architecture
        if self.architecture is None:
            #TODO: Implement default forward path
            x = F.relu(F.max_pool2d(self.conv1(x), kernel_size=3))
            x = F.relu(F.max_pool2d(self.conv2(x), kernel_size=2))
            x = F.relu(self.fc1(x.view(-1, 16)))
            x = self.fc2(x)

        # Linear fully connected architecture
        elif self.architecture == 'linear':
            x = self.layers(x.view(-1, 14**2))

        # ResNet architecture
        elif self.architecture == 'resnet':
            if self.batch_normalization:
                x = F.relu(self.bn(self.conv(x)))
            else:
                x = F.relu(self.conv(x))
            x = self.resnet_blocks(x)
            x = F.avg_pool2d(x, 32).view(x.size(0), -1)
            x = self.fc(x)

        elif architecture == 'lenet' or architecture == 'alexnet':
            raise NotImplementedError

        elif architecture == 'inception':
            raise NotImplementedError

        elif architecture == 'inceptionresnet':
            raise NotImplementedError

        elif architecture == 'xception':
            raise NotImplementedError

        else:
            raise NameError('Unknown architecture')

        return x

    def train(  self, \
                train_input, train_target, test_input = None, test_target = None, \
                batch_size = 10, epoch = 25, \
                eta = 1e-1, criterion = nn.CrossEntropyLoss(), print_skip = 5):
        '''
        Training method
        '''
        if self.optimizer is not None:
            if self.optimizer == 'SGD':
                optimizer = torch.optim.SGD(self.parameters(), lr = eta)
            elif self.optimizer == 'Adam':
                optimizer = torch.optim.Adam(self.parameters(), lr = eta)
            else:
                raise NameError('Unknown optimizer')
        for e in range(epoch):
            sum_loss = 0
            # We do this with mini-batches
            for b in range(0, train_input.size(0), batch_size):
                output = self(train_input.narrow(0, b, batch_size))
                loss = criterion(output, train_target.narrow(0, b, batch_size))
                sum_loss = sum_loss + loss.item()
                if self.optimizer is None:
                    self.zero_grad()
                else:
                    optimizer.zero_grad()
                loss.backward()
                if self.optimizer is None:
                    with torch.no_grad():
                        for p in self.parameters():
                            p -= eta * p.grad
                else:
                    optimizer.step()

            self.sumloss.append(sum_loss)

            self.train_error.append(self.compute_error_rate(train_input, train_target, batch_size))
            self.test_error.append(self.compute_error_rate(test_input, test_target, batch_size))

            if e%print_skip == 0:
                print("Epoch #{:d} --> Total train loss : {:.03f} ".format(e,  self.sumloss[-1]))
                print("------------> Train error rate : {:.02f}% ".format(self.train_error[-1]*100))
                if test_input is not None and test_target is not None:
                    print("------------> Test error rate : {:.02f}% ".format(self.test_error[-1]*100))
                print("--------------------------------------------")
            #Save best epoch
            if self.test_error[self.best_epoch] > self.test_error[-1]:
                self.best_epoch = e

        print("BEST SCORE --> Epoch #{:d}: train_error: {:.02f}%, test_error: {:.02f}%".format(self.best_epoch, self.train_error[self.best_epoch]*100, self.test_error[self.best_epoch]*100))

    def compute_error_rate(self, input, target, batch_size):
        '''
        Computing error rate givin an input and its target
        '''
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



class ResNetBlock(nn.Module):
    def __init__(self, nb_channels, kernel_size, skip_connections, batch_normalization):
        '''
        Initialization of a unit block of ResNet
        '''
        super(ResNetBlock, self).__init__()
        self.skip_connections = skip_connections
        self.batch_normalization = batch_normalization

        self.conv1 = nn.Conv2d(nb_channels, nb_channels,
                               kernel_size = kernel_size,
                               padding = (kernel_size - 1) // 2)
        if batch_normalization:
            self.bn1 = nn.BatchNorm2d(nb_channels)

        self.conv2 = nn.Conv2d(nb_channels, nb_channels,
                               kernel_size = kernel_size,
                               padding = (kernel_size - 1) // 2)
        if batch_normalization:
            self.bn2 = nn.BatchNorm2d(nb_channels)

    def forward(self, x):
        '''
        Forward path
        '''
        y = self.conv1(x)
        if self.batch_normalization:
            y = self.bn1(y)
        y = F.relu(y)
        y = self.conv2(y)
        if self.batch_normalization:
            y = self.bn2(y)
        if not self.skip_connections:
            y = y + x
        y = F.relu(y)

        return y
