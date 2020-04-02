import torch
from torch import nn
from utils import generate_pair_sets, plot_results
from Net import Net

import argparse

parser = argparse.ArgumentParser(description='Project 1 - Classification.')

parser.add_argument('--datasize',
                    type = int, default = 1000,
                    help = 'Number of pairs used for training and for testing (default: 1000)')

parser.add_argument('--architecture',
                    type = str, default = None,
                    help = 'Architecture of Neural Network to use (can be ????; default: ????)')

parser.add_argument('--loss',
                    type = str, default = None,
                    help = 'Loss used to train Neural Network (can be MSE, ????; default: crossentropy)')

parser.add_argument('--residual',
                    action='store_true', default=False,
                    help = 'Use residual Neural Network (default: False)')

parser.add_argument('--bn',
                    action='store_true', default=False,
                    help = 'Use batch normalization (default: False)')

parser.add_argument('--nodes',
                    type = int, default = 512,
                    help = 'Number of nodes (ignored if architecture is not linear; default: 512)')

parser.add_argument('--deep',
                    action='store_true', default=False,
                    help = 'Use deep Neural Network (ignored if architecture is not linear; default: False)')

parser.add_argument('--optimizer',
                    type = str, default = None,
                    help = 'Define optimizer to use (can be MSE, Adam; default: None)')

args = parser.parse_args()


## Determine loss used
if args.loss is None or args.loss == 'crossentropy':
    # Default loss
    loss = nn.CrossEntropyLoss()
elif args.loss == 'MSE':
    # MSE loss
    loss = nn.MSELoss()
    raise NotImplementedError
else:
    raise ValueError


## Data generation
train_input, train_target, train_classes, \
test_input, test_target, test_classes = generate_pair_sets(args.datasize)
print("** Data imported sucessfully **\n")

train_input = train_input.reshape(-1, 1, train_input.shape[-2], train_input.shape[-1])
test_input = test_input.reshape(-1, 1, test_input.shape[-2], test_input.shape[-1])
train_classes = train_classes.reshape(-1)
test_classes = test_classes.reshape(-1)

print("** Model choosen: **")

## Defining parameters
nb_residual_blocks = None
nb_channels = None
kernel_size = None
nb_linear_layers = None
nb_nodes = None

# Number of repetition
rep = 10
# Learning rate
eta = 1e-1
# Parameters for Neural Network
nb_classes = 10
if args.architecture == 'linear':
    nb_nodes = args.nodes
    if args.deep:
        nb_linear_layers = 3 # TO DEFINE

    else:
        nb_linear_layers = 1 # TO DEFINE

    print("*  Linear neural network with {} fully connected hidden layer.".format(nb_linear_layers))

elif args.architecture == 'resnet':
    nb_residual_blocks = 0 # TO DEFINE
    nb_channels = 0 # TO DEFINE
    kernel_size = 0 # TO DEFINE
    optimizer = 'SGD'
    print(  "*  Resnet architecture neural network with {} \
            residual block with {} channels and a kernel size of {}.".format(nb_residual_blocks, nb_channels, kernel_size))

elif args.architecture == 'lenet' or args.architecture == 'alexnet':
    raise NotImplementedError

elif args.architecture == 'inception':
    # Use batch normalization
    args.bn = True
    raise NotImplementedError

elif args.architecture == 'inceptionresnet':
    raise NotImplementedError

elif args.architecture == 'xception':
    raise NotImplementedError

else:
    args.architecture = None
    print("*  Default neural network architecture chosen.")


skip_connections = args.residual
if skip_connections:
    print("*  Skipping connections features activated!")

batch_normalization = args.bn
if skip_connections:
    print("*  Batch Normalization features activated!")

if args.optimizer is None:
    print("*  No optimizer choosen --> using batch stochastic gradient descend.")
elif args.optimizer == 'MSE':
    print("*  MSE Optimizer used.")
elif args.optimizer == 'Adam':
    print("*  Adam Optimizer used.")
else:
    raise ValueError("Unknown optimizer")


test_errors = []
train_errors = []
train_losses = []

for i in range(rep):
    ## Model declaration
    model = Net(args.architecture, nb_classes, nb_residual_blocks, \
                nb_channels, kernel_size, skip_connections, batch_normalization, \
                nb_linear_layers, nb_nodes, args.optimizer)
    print("\n** Model {} created sucessfully **\n".format(i+1))

    ## Model Training
    print("** Starting training... **")
    model.train(train_input, train_classes, test_input, test_classes, \
                eta = eta, criterion = loss)
    print("** Training done. **\n")

    ## Results saving
    test_errors.append(model.test_error)
    train_errors.append(model.train_error)
    train_losses.append(model.sumloss)
    print("**************************************************************")

## Ploting results
plot_results(train_losses, train_errors, test_errors)
