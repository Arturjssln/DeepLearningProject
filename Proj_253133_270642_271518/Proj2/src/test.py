import torch
from utils import generate_data, plot_results
from net import Net
import framework as ff

torch.set_grad_enabled(False)

DATASET_SIZE = 1000

## Generate dataset
train_input, train_target, test_input, test_target = generate_data(DATASET_SIZE)

## Create Model
model = Net(nb_nodes = 25)
print(model)
## Training model
model.train(train_input, train_target, test_input, test_target)
## Ploting results
plot_results(model.sumloss, model.train_error, model.test_error)
