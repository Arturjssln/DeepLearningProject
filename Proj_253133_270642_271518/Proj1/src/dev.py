import torch
from torch import nn
from utils import *
from Net import Net

parser = argparse.ArgumentParser(description='Project 1 - Classification.')

parser.add_argument('--datasize',
                    type = int, default = 1000,
                    help = 'Number of pairs used for training and for testing (default: 1000)')

parser.add_argument('--architecture',
                    type = str, default = None,
                    help = 'Architecture of Neural Network to use (can be ????; default: ????)')

parser.add_argument('--loss',
                    type = str, default = None,
                    help = 'Loss used to train Neural Network (can be MSE, ????; default: MSE)')

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


args = parser.parse_args()

if args.loss is None:
    loss = nn.CrossEntropyLoss()
else:
    if args.loss == "MSE":
        loss = nn.MSELoss()
    else:
        raise ValueError

## Data generation
train_input, train_target, train_classes, \
test_input, test_target, test_classes = generate_pair_sets(args.datasize)
print("# Data imported sucessfully\n")

train_input = train_input.reshape(-1, 1, train_input.shape[-2], train_input.shape[-1])
test_input = test_input.reshape(-1, 1, test_input.shape[-2], test_input.shape[-1])
train_classes = train_classes.reshape(-1)
test_classes = test_classes.reshape(-1)

## Defining parameters
nb_residual_blocks = None
nb_channels = None
kernel_size = None
nb_linear_layers = None
nb_nodes = None
# Number of repetition
rep = 10
# parameters for Neural Network
nb_classes = 10

if args.architecture is 'linear':
    nb_nodes = args.nodes
    if args.deep:
        nb_linear_layers = 5 # TO DEFINE
    else:
        nb_linear_layers = 1 # TO DEFINE

if args.architecture is 'resnet':
    nb_residual_blocks = 0 # TO DEFINE
    nb_channels = 0 # TO DEFINE
    kernel_size = 0 # TO DEFINE

skip_connections = args.residual
batch_normalization = args.bn


test_errors = []
train_errors = []
train_losses = []

for i in range(rep):
    ## Model declaration

    model = Net(args.architecture, nb_classes, nb_residual_blocks, \
                nb_channels, kernel_size, skip_connections, batch_normalization, \
                nb_linear_layers, nb_nodes)
    print("# Model {} created sucessfully\n".format(i))

    ## Model Training
    print("# Starting training...\n")
    model.train(train_input, train_classes, test_input, test_classes, \
                eta = 1e-1, criterion = loss)
    print("# Training done.\n")

    ## Results saving
    test_errors.append(model.test_error)
    train_errors.append(model.train_error)
    train_losses.append(model.sumloss)

## Ploting results
plot_results(train_losses, train_errors, test_errors)
