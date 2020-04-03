import torch
from torch import nn
from utils import generate_pair_sets, plot_results, count_parameters
from Net import Net
import time
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
                    type = int, default = 32,
                    help = 'Number of nodes (ignored if architecture is not linear; default: 32)')

parser.add_argument('--epoch',
                    type = int, default = 25,
                    help = 'Number of epoch (default: 25)')

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
test_input, test_target, test_classes = generate_pair_sets(args.datasize, normalize = True)
print("** Data imported sucessfully **\n")
print(train_input.shape)
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

    print("*  Linear neural network with {} fully connected hidden layer with {} nodes.".format(nb_linear_layers, nb_nodes))

elif args.architecture == 'resnet':
    nb_residual_blocks = 1 # TO DEFINE
    nb_channels = 6 # TO DEFINE
    kernel_size = 3 # TO DEFINE but minimum 3 (included)
    optimizer = 'SGD'
    print(  "*  Resnet architecture neural network with {} residual block with {} channels and a kernel size of {}.".format(nb_residual_blocks, nb_channels, kernel_size))

elif args.architecture == 'lenet' or args.architecture == 'alexnet':
    args.optimizer = 'SGD'
    print("*  LeNet neural network.")

elif args.architecture == 'xception':
    args.bn = True
    print("*  Xception neural network.")

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
elif args.optimizer == 'SGD':
    print("*  SGD Optimizer used.")
elif args.optimizer == 'Adam':
    print("*  Adam Optimizer used.")
else:
    raise ValueError("Unknown optimizer")


goal_errors = []
test_errors = []
train_errors = []
train_losses = []

for i in range(rep):
    start_rep_time = time.time()
    ## Model declaration
    model = Net(args.architecture, nb_classes, nb_residual_blocks, \
                nb_channels, kernel_size, skip_connections, batch_normalization, \
                nb_linear_layers, nb_nodes, args.optimizer)
    print("\n** Model {} created sucessfully **\n".format(i+1))
    print("** Model has {} parameters **".format(count_parameters(model)))
    ## Model Training
    print("** Starting training... **")
    model.train(train_input, train_classes, test_input, test_classes, test_target, \
                epoch = args.epoch, eta = eta, criterion = loss)
    
    print("** Training done. **\n")

    ## Results saving
    goal_errors.append(model.test_final_error)
    test_errors.append(model.test_error)
    train_errors.append(model.train_error)
    train_losses.append(model.sumloss)
    print("** Training time : {:.0f} minutes {:.0f} seconds".format((time.time()-start_rep_time)/60, (time.time()-start_rep_time)%60))
    print("**************************************************************")

## Ploting results
plot_results(train_losses, train_errors, test_errors, goal_errors)
