from net import Net
import framework as ff
from utils import generate_data, plot_results

## Generate dataset
DATASET_SIZE = 1000
train_input, train_target, test_input, test_target = generate_data(DATASET_SIZE, one_hot=False)

model = Net(nb_nodes=25)

## Training model
model.train(train_input, train_target, test_input, test_target, epoch=100, eta=1e-1, criterion=ff.CrossEntropyLoss())

## Ploting results
plot_results(model.sumloss, model.train_error, model.test_error)
