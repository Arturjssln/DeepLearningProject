import torch.empty
from utils import generate_data, plot_results
from Net import Net

torch.set_grad_enabled(False)

dataset_size = 1000

## Generate dataset
train_input, train_target, test_input, test_target = generate_data(dataset_size)

## Create Model
model = Net(nb_nodes = 25)

## Training model
model.train(train_input, train_target, test_input, test_target):

## Ploting results
plot_results(model.sumloss, model.train_error, model.test_error)
