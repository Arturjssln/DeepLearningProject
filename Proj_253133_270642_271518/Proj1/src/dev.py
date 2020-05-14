
import torch
from torch import nn
from utils import generate_data, plot_results, count_parameters
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

parser.add_argument('--dropout',
                    action='store_true', default=False,
                    help = 'Use dropout (default: False)')

parser.add_argument('--save_fig',
                    action='store_true', default=False,
                    help = 'Save figure to png')

parser.add_argument('--nb_residual_blocks',
                    type = int, default = 1, #To Define
                    help = 'If Resnet selected - change number of residual blocks (default is 1)')

parser.add_argument('--nb_channels',
                    type = int, default = 6, #To Define
                    help = 'If Resnet selected - change number of channels (default is 6)')

parser.add_argument('--kernel_size',
                    type = int, default = 3, #To Define
                    help = 'If Resnet selected - change kernel size (default is 3)')

parser.add_argument('--force_axis',
                    action='store_true', default=False,
                    help = 'Used for plotting, if selected, axis is not automatically scalled (default is false)')

parser.add_argument('--auxloss',
                    action='store_true', default=False,
                    help = 'Use auxiliary loss (default is false)')


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


print("** Model chosen: **")

## Defining parameters
nb_residual_blocks = None
nb_channels = None
kernel_size = None
nb_linear_layers = None
nb_nodes = None

# Number of repetition

rep = 20

# Learning rate
eta = 1e-1
# Parameters for Neural Network
nb_classes = 10
if args.architecture == 'linear':
    nb_nodes = args.nodes
    args.auxloss = False
    if args.deep:
        nb_linear_layers = 5
    else:
        nb_linear_layers = 3

    print("*  Linear neural network with {} fully connected hidden layer with {} nodes.".format(nb_linear_layers, nb_nodes))

elif args.architecture == 'resnet':
    nb_residual_blocks = args.nb_residual_blocks
    nb_channels = args.nb_channels
    kernel_size = args.kernel_size
    args.auxloss = False

    ## Kernel size must be greater than 2
    if kernel_size < 3:
        print("*  WARNING KERNEL SIZE MUST BE GREATER THAN 2, CHANGED KERNEL SIZE TO 3")
        kernel_size = 3

    optimizer = 'SGD'
    print(  "*  Resnet architecture neural network with {} residual block with {} channels and a kernel size of {}.".format(nb_residual_blocks, nb_channels, kernel_size))

elif args.architecture == 'lenet':
    kernel_size = args.kernel_size # CAN BE 3 OR 5
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
    args.architecture = None
    print("*  Default neural network architecture chosen.")


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


goal_errors = []
test_errors = []
train_errors = []
train_losses = []

for i in range(rep):
    start_rep_time = time.time()

    ## Model declaration
    model = Net(args.architecture, nb_classes, nb_residual_blocks, \
                nb_channels, kernel_size, skip_connections, args.bn, \
                nb_linear_layers, nb_nodes, args.optimizer, args.dropout, args.auxloss)
    print("** Model {} created sucessfully **".format(i+1))
    print("** Model has {} parameters **\n".format(count_parameters(model)))

    ## Data generation
    train_input, train_target, train_classes, \
    test_input, test_target, test_classes = generate_data(args.datasize, normalize = True)


    ## Model Training
    print("** Starting training... **")
    success = model.train_(   train_input, train_classes, test_input, test_classes, test_target, \
                    epoch = args.epoch, eta = eta, criterion = loss)
    print("** Training done. **")

    ## Results saving
    if success:
        goal_errors.append(model.test_final_error)
        test_errors.append(model.test_error)
        train_errors.append(model.train_error)
        train_losses.append(model.sumloss)

        print("** Training time : {:.0f} minutes {:.0f} seconds\n".format((time.time()-start_rep_time)//60, int(time.time()-start_rep_time)%60))
    else:
        print('** Training failed. **')
    print("**************************************************************")

## Ploting results
#DEFAULT
plot_results(train_losses, train_errors, test_errors, goal_errors, args.force_axis)
#RESNET
#plot_results(train_losses, train_errors, test_errors, goal_errors, args.force_axis, args.save_fig, "Resnet-{} channels-{} residual-{} kernelsize".format(nb_channels, nb_residual_blocks, kernel_size))
#LENET
#plot_results(train_losses, train_errors, test_errors, goal_errors, args.force_axis, args.save_fig, "Lenet-Kernelsize {}-BatchNorm {}-Dropout {}-Aux loss {}".format(kernel_size, args.bn, args.dropout, args.auxloss))
#LINEAR
#plot_results(train_losses, train_errors, test_errors, goal_errors, args.force_axis, args.save_fig, "Linear-{} linear_layers-{} nodes-Dropout {}".format(nb_linear_layers, nb_nodes, args.dropout))
