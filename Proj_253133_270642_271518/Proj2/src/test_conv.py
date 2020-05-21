import torch
from utils import load_data, plot_results, plot_prediction_mnist
from net_conv import Net
import framework as ff
import argparse
import matplotlib.pyplot as plt

torch.set_grad_enabled(False)

parser = argparse.ArgumentParser(description='Project 2 - NN Framework.')
parser.add_argument('--train',
                    action='store_true', default=False,
                    help='Train the neural network')
args = parser.parse_args()

ONE_HOT = False
loss = ff.CrossEntropyLoss()

DATASET_SIZE = 100

## Generate dataset
train_input, train_target, test_input, test_target = load_data(nb_data=DATASET_SIZE, one_hot_labels=ONE_HOT, normalize=True)

## Create model
model = Net()
print(model)
if args.train:
    print('Using Cross Entropy Loss\n')
    ## Training model
    torch.manual_seed(3)
    model.train_(train_input, train_target, test_input, test_target, epoch=5, eta=1e-1, criterion=loss)
    ## Ploting results of model at the end of training
    plot_results(model.sumloss, model.train_error, model.test_error)
    plot_prediction_mnist(test_input, test_target, model)
    plt.suptitle('Prediction of the model at the end of training')
    plt.show()
else:
    ## Load best model
    model.load('../model/best-model-conv.pt')
    ## Ploting results of best model
    plot_prediction_mnist(test_input, test_target, model)
    plt.suptitle('Prediction of the best model')
    plt.show()
