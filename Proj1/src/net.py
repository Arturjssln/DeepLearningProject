import math
import torch
from torch import nn
from torch.nn import functional as F
import time

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
        self.test_final_error = []
        self.train_error = []
        self.sumloss = []
        self.best_score = []
        self.best_epoch = 0
        self.architecture = architecture

        # Usefull parameters will have values, other will be None
        nb_classes, nb_residual_blocks, \
        nb_channels, kernel_size, \
        skip_connections, batch_normalization, \
        nb_linear_layers, nb_nodes, optimizer, \
        self.dropout, self.auxloss = param

        self.optimizer = optimizer

        # default architecture
        if architecture is None:
            raise NotImplementedError

        # Linear fully connected architecture
        elif architecture == 'linear':
            modules = []
            modules.append(nn.Linear(14**2, nb_nodes))
            modules.append(nn.ReLU())
            for _ in range(nb_linear_layers - 1):
                modules.append(nn.Linear(nb_nodes, nb_nodes))
                modules.append(nn.ReLU())
                if(self.dropout):
                    modules.append(nn.Dropout()) 
            modules.append(nn.Linear(nb_nodes, nb_classes))
            self.layers = nn.Sequential(*modules)

        # ResNet architecture
        elif architecture == 'resnet':
            self.do = nn.Dropout()
            self.batch_normalization = batch_normalization
            self.conv = nn.Conv2d(1, nb_channels, kernel_size = kernel_size, padding = (kernel_size - 1) // 2)
            if batch_normalization:
                self.bn = nn.BatchNorm2d(nb_channels)
            self.resnet_blocks = nn.Sequential(
                *(ResNetBlock(nb_channels, kernel_size, skip_connections, batch_normalization, self.dropout) for _ in range(nb_residual_blocks))
            )
            self.fc = nn.Linear(nb_channels, nb_classes)


        # LeNet architecture
        elif architecture == 'lenet':
            self.batch_normalization = batch_normalization
            self.do = nn.Dropout()

            if batch_normalization:
                if kernel_size == 3:
                    self.c1 = nn.Sequential(nn.Conv2d(1, 6, kernel_size=kernel_size, padding = 2), nn.BatchNorm2d(6), nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2))
                elif kernel_size == 5:
                    self.c1 = nn.Sequential(nn.Conv2d(1, 6, kernel_size=kernel_size, padding = 2), nn.BatchNorm2d(6), nn.ReLU())
                else:
                    raise NameError("Kernel size not valid (Can be 3 or 5)")
                self.c2 = nn.Sequential(nn.Conv2d(6, 16, kernel_size=kernel_size), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2))
                if self.auxloss:
                    self.fc_auxloss = nn.Linear(16*kernel_size*kernel_size, nb_classes)
                self.c3 = nn.Sequential(nn.Conv2d(16, 120, kernel_size=kernel_size), nn.BatchNorm2d(120), nn.ReLU())
                self.fc = nn.Sequential(nn.Linear(120, 84), nn.ReLU(), nn.Linear(84, nb_classes))
            else:
                if kernel_size == 3:
                    self.c1 = nn.Sequential(nn.Conv2d(1, 6, kernel_size=kernel_size, padding = 2), nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2))
                elif kernel_size == 5:
                    self.c1 = nn.Sequential(nn.Conv2d(1, 6, kernel_size=kernel_size, padding = 2), nn.ReLU())
                else:
                    raise NameError("Kernel size not valid (Can be 3 or 5)")
                self.c2 = nn.Sequential(nn.Conv2d(6, 16, kernel_size=kernel_size), nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2))
                if self.auxloss:
                    self.fc_auxloss = nn.Linear(16*kernel_size*kernel_size, nb_classes)
                self.c3 = nn.Sequential(nn.Conv2d(16, 120, kernel_size=kernel_size), nn.ReLU())
                self.fc = nn.Sequential(nn.Linear(120, 84), nn.ReLU(), nn.Linear(84, nb_classes))


        # AlexNet architecture
        elif architecture == 'alexnet':
            #AlexCONV3(256,384,k=3,s=1,p=1)
            #AlexCONV4(384, 384, k=3,s=1,p=1)
            #AlexCONV5(384, 256, k=3, s=1,p=1)
            #AlexFC6(256*6*6, 4096)
            #AlexFC6(4096,4096)
            #AlexFC6(4096,1000)
            self.c1 = nn.Sequential(nn.Conv2d(1, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2))
            self.c2 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1), nn.ReLU(), nn.MaxPool2d(kernel_size=2,stride=2))
            self.c3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
            if self.auxloss:
                self.bn_auxloss = nn.BatchNorm2d(128)
                self.fc_auxloss = nn.Linear(128*3*3, nb_classes)
            self.c4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
            self.c5 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1), nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2))
            self.fc = nn.Sequential(nn.Linear(256, 1024), nn.ReLU(), nn.Linear(1024, 512), nn.ReLU(), nn.Linear(512, nb_classes))

        elif architecture == 'xception':
            self.conv1 = nn.Conv2d(1, 32, 3, 2, 0, bias=False)
            self.bn1 = nn.BatchNorm2d(32)
            self.relu = nn.ReLU(inplace=True)
            self.conv2 = nn.Conv2d(32,64,3,bias=False)
            self.bn2 = nn.BatchNorm2d(64)
            self.block1=XceptionBlock(64,128,2,2,start_with_relu=False,grow_first=True)
            self.block2=XceptionBlock(128,256,2,2,start_with_relu=True,grow_first=True)
            self.block3=XceptionBlock(256,728,2,2,start_with_relu=True,grow_first=True)
            self.block4=XceptionBlock(728,728,3,1,start_with_relu=True,grow_first=True)
            self.block5=XceptionBlock(728,728,3,1,start_with_relu=True,grow_first=True)
            self.block6=XceptionBlock(728,728,3,1,start_with_relu=True,grow_first=True)
            self.block7=XceptionBlock(728,728,3,1,start_with_relu=True,grow_first=True)
            self.block8=XceptionBlock(728,728,3,1,start_with_relu=True,grow_first=True)
            self.block9=XceptionBlock(728,728,3,1,start_with_relu=True,grow_first=True)
            self.block10=XceptionBlock(728,728,3,1,start_with_relu=True,grow_first=True)
            self.block11=XceptionBlock(728,728,3,1,start_with_relu=True,grow_first=True)
            self.block12=XceptionBlock(728,1024,2,2,start_with_relu=True,grow_first=False)
            self.conv3 = SeparableConv2d(1024,1536,3,1,1)
            self.bn3 = nn.BatchNorm2d(1536)
            self.conv4 = SeparableConv2d(1536,2048,3,1,1)
            self.bn4 = nn.BatchNorm2d(2048)
            self.fc = nn.Linear(2048, nb_classes)
            #------- init weights --------
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, math.sqrt(2. / n))
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
            #-----------------------------

        else:
            raise NameError('Unknown architecture')

    def forward(self, x):
        '''
        Forward path
        '''
        # default architecture
        if self.architecture is None:
            raise NotImplementedError

        # Linear fully connected architecture
        elif self.architecture == 'linear':
            x = self.layers(x.view(-1, 14**2))

        # ResNet architecture
        elif self.architecture == 'resnet':
            batch_size = x.size(0)
            if self.batch_normalization:
                if self.dropout:
                    x = F.relu(self.bn(self.do(self.conv(x))))
                else:
                    x = F.relu(self.bn(self.conv(x)))
            else:
                x = F.relu(self.conv(x))
            x = self.resnet_blocks(x)
            x = F.avg_pool2d(x, 14).view(batch_size, -1)
            x = self.fc(x)

        # LeNet architecture
        elif self.architecture == 'lenet':
            batch_size = x.size(0)
            if self.dropout:
                x = self.do(self.c1(x))
                x = self.do(self.c2(x))
                if self.auxloss:
                    aux = self.do(self.fc_auxloss(x.view(batch_size, -1)))
                x = self.do(self.c3(x))
                x = self.do(self.fc(x.view(batch_size, -1)))
            else:
                x = self.c1(x)
                x = self.c2(x)
                if self.auxloss:
                    aux = self.fc_auxloss(x.view(batch_size, -1))
                x = self.c3(x)
                x = self.fc(x.view(batch_size, -1))


        # AlexNet architecture
        elif self.architecture == 'alexnet':
            batch_size = x.size(0)
            x = self.c1(x)
            x = self.c2(x)
            x = self.c3(x)
            x = self.c4(x)
            x = self.c5(x)
            x = self.fc(x.view(batch_size, -1))

        elif self.architecture == 'xception':
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.conv2(x)
            x = self.bn2(x)
            x = self.relu(x)
            x = self.block1(x)
            x = self.block2(x)
            x = self.block3(x)
            x = self.block4(x)
            x = self.block5(x)
            x = self.block6(x)
            x = self.block7(x)
            x = self.block8(x)
            x = self.block9(x)
            x = self.block10(x)
            x = self.block11(x)
            x = self.block12(x)
            x = self.conv3(x)
            x = self.bn3(x)
            x = self.relu(x)
            x = self.conv4(x)
            x = self.bn4(x)
            x = self.relu(x)
            x = F.adaptive_avg_pool2d(x, (1, 1))
            x = x.view(x.size(0), -1)
            x = self.fc(x)

        else:
            raise NameError('Unknown architecture')

        if self.auxloss:
            return x, aux
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def train_( self, \
                train_input, train_target, \
                test_input = None, test_target = None, test_target_final = None, \
                batch_size = 10, epoch = 50, \
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

        train_time_avg_epoch = 0

        for e in range(epoch):
            epoch_start_time = time.time()
            sum_loss = 0
            # We do this with mini-batches
            for b in range(0, train_input.size(0), batch_size):

                if self.auxloss:
                    output, aux_output = self(train_input.narrow(0, b, batch_size))
                    loss = criterion(output, train_target.narrow(0, b, batch_size))
                    loss2 = criterion(aux_output, train_target.narrow(0, b, batch_size))
                    sum_loss = sum_loss + 0.7 * loss.item() + 0.3 * loss2.item()
                else:
                    output = self(train_input.narrow(0, b, batch_size))
                    loss = criterion(output, train_target.narrow(0, b, batch_size))
                    sum_loss = sum_loss + loss.item()


                if self.optimizer is None:
                    self.zero_grad()
                else:
                    optimizer.zero_grad()
                if self.auxloss:
                    loss2.backward(retain_graph = True)
                loss.backward()
                if self.optimizer is None:
                    with torch.no_grad():
                        for p in self.parameters():
                            p -= eta * p.grad
                else:
                    optimizer.step()
            if math.isnan(sum_loss):
                return False
            self.sumloss.append(sum_loss)

            self.train_error.append(self.compute_error_rate(train_input, train_target, batch_size))
            self.test_error.append(self.compute_error_rate(test_input, test_target, batch_size))
            self.test_final_error.append(self.compute_error_rate(test_input, test_target_final, batch_size, pair = True))

            train_time_avg_epoch = ((train_time_avg_epoch * e) + (time.time() - epoch_start_time)) / (e+1)
            remaining_time  = (epoch - e) * train_time_avg_epoch

            if e%print_skip == 0:
                print("Epoch #{:2d} --> Total train loss : {:.03f} ".format(e,  self.sumloss[-1]))
                print("------------> Train error rate : {:.02f}% ".format(self.train_error[-1]*100))
                if test_input is not None and test_target is not None and test_target_final is not None:
                    print("------------> Test class error rate : {:.02f}% ".format(self.test_error[-1]*100))
                    print("------------> Test comparison error rate : {:.02f}% ".format(self.test_final_error[-1]*100))
                print("Predicted remaining time : {:.0f} minute(s) {:.0f} second(s)".format(remaining_time//60, int(remaining_time)%60))
                print("-------------------------------------------------")

            #Save best epoch
            if self.test_error[self.best_epoch] > self.test_error[-1]:
                self.best_epoch = e

        self.best_score = [self.train_error[self.best_epoch]*100, self.test_error[self.best_epoch]*100,  self.test_final_error[self.best_epoch]*100]
        print("** BEST SCORE --> Epoch #{:d}: \n*  train_error: {:.02f}%, \n*  test_error: {:.02f}%, \n*  test_comparison_error: {:.02f}%"\
            .format(self.best_epoch, self.train_error[self.best_epoch]*100, self.test_error[self.best_epoch]*100, self.test_final_error[self.best_epoch]*100))
        return True

    def compute_error_rate(self, input, target, batch_size, pair = False):
        '''
        Computing error rate givin an input and its target
        '''
        self.eval()
        error = 0.0
        for b in range(0, target.size(0), batch_size):
            if pair:
                if self.auxloss:
                    prediction, _ = self(input.narrow(0, 2*b, 2*batch_size))
                else:
                    prediction = self(input.narrow(0, 2*b, 2*batch_size))
                _, prediction = prediction.max(1)
                prediction = prediction.reshape(-1, 2)
                predicted_classes = (prediction[:, 1] - prediction[:, 0]) >= 0
                predicted_classes = predicted_classes.int()
            else:
                if self.auxloss:
                    prediction, _ = self(input.narrow(0, b, batch_size))
                else:
                    prediction = self(input.narrow(0, b, batch_size))
                _, predicted_classes = prediction.max(1)
            # Calculate test error
            for pred, t in zip(predicted_classes, target.narrow(0, b, batch_size)):
                if pred.item() != t.item():
                    error += 1
        error /= target.size(0)
        self.train()
        return error



class ResNetBlock(nn.Module):
    def __init__(self, nb_channels, kernel_size, skip_connections, batch_normalization, dropout):
        '''
        Initialization of a unit block of ResNet
        '''
        super(ResNetBlock, self).__init__()
        self.skip_connections = skip_connections
        self.batch_normalization = batch_normalization
        self.dropout = dropout
        self.do = nn.Dropout()
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
        if self.dropout:
            y = self.do(y)
        y = F.relu(y)
        y = self.conv2(y)
        if self.batch_normalization:
            y = self.bn2(y)
        if self.dropout:
            y = self.do(y)
        if not self.skip_connections:
            y = y + x
        y = F.relu(y)
        return y


class SeparableConv2d(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=1,stride=1,padding=0,dilation=1,bias=False):
        super(SeparableConv2d,self).__init__()

        self.conv1 = nn.Conv2d(in_channels,in_channels,kernel_size,stride,padding,dilation,groups=in_channels,bias=bias)
        self.pointwise = nn.Conv2d(in_channels,out_channels,1,1,0,1,1,bias=bias)

    def forward(self,x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x


class XceptionBlock(nn.Module):
    def __init__(self,in_filters,out_filters,reps,strides=1,start_with_relu=True,grow_first=True):
        super(XceptionBlock, self).__init__()

        if out_filters != in_filters or strides!=1:
            self.skip = nn.Conv2d(in_filters,out_filters,1,stride=strides, bias=False)
            self.skipbn = nn.BatchNorm2d(out_filters)
        else:
            self.skip=None

        self.relu = nn.ReLU(inplace=True)
        rep=[]

        filters=in_filters
        if grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters,out_filters,3,stride=1,padding=1,bias=False))
            rep.append(nn.BatchNorm2d(out_filters))
            filters = out_filters

        for i in range(reps-1):
            rep.append(self.relu)
            rep.append(SeparableConv2d(filters,filters,3,stride=1,padding=1,bias=False))
            rep.append(nn.BatchNorm2d(filters))

        if not grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters,out_filters,3,stride=1,padding=1,bias=False))
            rep.append(nn.BatchNorm2d(out_filters))

        if not start_with_relu:
            rep = rep[1:]
        else:
            rep[0] = nn.ReLU(inplace=False)

        if strides != 1:
            rep.append(nn.MaxPool2d(3,strides,1))
        self.rep = nn.Sequential(*rep)

    def forward(self,inp):
        x = self.rep(inp)

        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp

        x+=skip
        return x
