import torch
from torch import nn
from utils import *
from Net import Net

parser = argparse.ArgumentParser(description='Project 1 - Classification.')

parser.add_argument('--datasize',
                    type = int, default = 1000,
                    help = 'Number of pairs used for training and for testing (default = 1000)')

parser.add_argument('--architecture',
                    type = str, default = None,
                    help = 'Architecture of Neural Network to use (can be ????; default = ????)')

parser.add_argument('--loss',
                    type = str, default = None,
                    help = 'Loss used to train Neural Network (can be MSE, ????; default = MSE)')

parser.add_argument('--residual',
                    action='store_true', default=False,
                    help = 'Use residual Neural Network (default False)')

parser.add_argument('--bn',
                    action='store_true', default=False,
                    help = 'Use batch normalization (default False)')

args = parser.parse_args()

if agrs.loss is None:
    loss == nn.MSELoss()
else:
    if args.loss == "MSE":
        loss == nn.MSELoss()
    else:
        raise ValueError



## Data generation
train_input, train_target, train_classes, \
test_input, test_target, test_classes = generate_pair_sets(args.datasize)
print("# Data imported sucessfully\n")

## Model generation
model = Net(architecture = args.architecture, skip_connections = args.residual, batch_normalization = agrs.bn)
print("# Model created sucessfully\n")

## Model training
print("# Starting training...\n")
model.train(train_input, train_classes, test_input, test_classes, \
            eta = 1e-1, criterion = loss)
print("# Training done.\n")

## Ploting results
plot_results(model.sumloss, model.train_error, model.test_error)
