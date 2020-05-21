
import torch
from torch import nn
from utils import generate_data, plot_training, count_parameters
from net import Net
import time
import argparse


parser = argparse.ArgumentParser(description='Project 1 - Classification.')

parser.add_argument('--datasize',
                    type=int, default=1000,
                    help='Number of pairs used for training and for testing; default: 1000')

parser.add_argument('--architecture',
                    type=str, default='resnet',
                    help='Architecture of Neural Network to use (can be linear, lenet, resnet, alexnet; default: resnet)')

parser.add_argument('--residual',
                    action='store_true', default=False,
                    help='Use residual Neural Network')

parser.add_argument('--bn',
                    action='store_true', default=False,
                    help='Use batch normalization')

parser.add_argument('--nodes',
                    type=int, default=32,
                    help='Number of nodes (ignored if architecture is not linear; default: 32)')

parser.add_argument('--epoch',
                    type=int, default=25,
                    help='Number of epoch (default: 25)')

parser.add_argument('--optimizer',
                    type=str, default=None,
                    help='Define optimizer to use (can be MSE, Adam; default: None)')

parser.add_argument('--dropout',
                    action='store_true', default=False,
                    help='Use dropout')

parser.add_argument('--nb_residual_blocks',
                    type=int, default=2,
                    help='If Resnet selected - change number of residual blocks (default is 2)')

parser.add_argument('--nb_channels',
                    type=int, default=16,
                    help='If Resnet selected - change number of channels (default is 16)')

parser.add_argument('--kernel_size',
                    type=int, default=7,
                    help='If Resnet selected - change kernel size (default is 7)')

parser.add_argument('--auxloss',
                    action='store_true', default=False,
                    help='Use auxiliary loss')


args = parser.parse_args()



print("** Model chosen: **")

## Defining parameters
nb_residual_blocks = None
nb_channels = None
kernel_size = None
nb_linear_layers = None
nb_nodes = None
loss=nn.CrossEntropyLoss()

# Number of repetition
rep = 20

# Learning rate
eta = 1e-1
# Parameters for Neural Network
nb_classes = 10
if args.architecture == 'linear':
    nb_nodes = args.nodes
    args.auxloss = False
    nb_linear_layers = 3

    print("*  Linear neural network with {} fully connected hidden layer with {} nodes.".format(nb_linear_layers, nb_nodes))

elif args.architecture == 'resnet':
    nb_residual_blocks = args.nb_residual_blocks
    nb_channels = args.nb_channels
    kernel_size = args.kernel_size
    args.auxloss = False

    ## Kernel size must be greater than 2
    if kernel_size < 3:
        kernel_size = 3
    args.optimizer = 'SGD'
    print("*  Resnet architecture neural network with {} residual block(s) with {} channel(s) and a kernel size of {}.".format(
        nb_residual_blocks, nb_channels, kernel_size))

elif args.architecture == 'lenet':
    kernel_size = args.kernel_size  # CAN BE 3 OR 5
    args.optimizer = 'SGD'
    print("*  LeNet neural network.")

elif args.architecture == 'alexnet':
    args.auxloss = False
    args.optimizer = 'SGD'
    print("*  AlexNet neural network.")


elif args.architecture == 'xception':
    args.bn = True
    print("*  Xception neural network.")

else:
    raise ValueError('Unkwown architecture')


skip_connections = args.residual
if skip_connections:
    print("*  Skipping connections feature activated!")

if args.bn:
    print("*  Batch Normalization feature activated!")

if args.dropout:
    print("*  Dropout feature activated!")

if args.auxloss:
    print("*  Auxiliary loss feature activated!")


if args.optimizer is None:
    print("*  No optimizer choosen --> using batch stochastic gradient descend.")
elif args.optimizer == 'SGD':
    print("*  SGD Optimizer used.")
elif args.optimizer == 'Adam':
    print("*  Adam Optimizer used.")
else:
    raise ValueError("Unknown optimizer")

start_rep_time = time.time()

## Model declaration
model = Net(args.architecture, nb_classes, nb_residual_blocks,
            nb_channels, kernel_size, skip_connections, args.bn,
            nb_linear_layers, nb_nodes, args.optimizer, args.dropout, args.auxloss)
print("** Model created sucessfully **")
print("** Model has {} parameters **".format(count_parameters(model)))

## Data generation
train_input, train_target, train_classes, \
    test_input, test_target, test_classes = generate_data(
        args.datasize, normalize=True)

## Model Training
print("** Starting training... **")
torch.manual_seed(0)
SUCCESS = model.train_(train_input, train_classes, test_input, test_classes, test_target,
                        epoch=args.epoch, eta=eta, criterion=loss)

## Results saving
if SUCCESS:
    print("** Training done in : {:.0f} minutes {:.0f} seconds\n".format(
        (time.time()-start_rep_time)//60, int(time.time()-start_rep_time) % 60))
else:
    print('** Training failed. **')

## Ploting results
plot_training(model.sumloss, model.train_error, model.test_error, model.test_final_error)
