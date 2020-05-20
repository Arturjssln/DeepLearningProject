import torch
from utils import generate_data, plot_results, plot_prediction
from net import Net
import framework as ff
import argparse
import matplotlib.pyplot as plt

torch.set_grad_enabled(False)

parser = argparse.ArgumentParser(description='Project 2 - NN Framework.')
parser.add_argument('--loss',
                    type=str, default='MSE',
                    help='Loss to use (available: CrossEntropy, MSE; default: MSE)')
args = parser.parse_args()
if args.loss == 'MSE':
    one_hot = True
    loss = ff.MSELoss()
elif args.loss == 'CrossEntropy':
    one_hot = False
    loss = ff.CrossEntropyLoss()
else:
    raise ValueError('Unknown loss.')

DATASET_SIZE = 1000

## Generate dataset
train_input, train_target, test_input, test_target = generate_data(DATASET_SIZE, one_hot=one_hot, normalize=True)

## Create model
model = Net(nb_nodes = 25)
print(model)
print('Using : {}Loss\n'.format(args.loss))
## Training model
model.train(train_input, train_target, test_input, test_target, epoch=100, eta = 1e-1, criterion=loss)
## Ploting results of model at the end of training
plot_results(model.sumloss, model.train_error, model.test_error)
plot_prediction(test_input, test_target, model)
plt.suptitle('Prediction of the model at the end of training')

## Load best model
model.load()
## Ploting results of best model
plot_prediction(test_input, test_target, model)
plt.suptitle('Prediction of the best model')
plt.show()
