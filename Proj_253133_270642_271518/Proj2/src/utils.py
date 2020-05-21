import math
import torch 
import matplotlib.pyplot as plt
from torchvision import datasets

data_dir = '../data'

def generate_set(nb, one_hot):
    input = torch.empty(nb, 2).uniform_(0, 1)
    target = input.pow(2).sum(dim = 1).sub(1 / (2*math.pi)).sign().sub(-1).div(2).long()
    if one_hot:
        target = convert_to_one_hot_labels(input, target)
    return input, target

def generate_data(nb, normalize=False, one_hot=True):
    train_input, train_target = generate_set(nb, one_hot)
    test_input, test_target = generate_set(nb, one_hot)

    if normalize:
        mu, std = train_input.mean(), train_input.std()
        train_input.sub_(mu).div_(std)
        test_input.sub_(mu).div_(std)

    return train_input, train_target, test_input, test_target

def convert_to_one_hot_labels(input, target):
    tmp = input.new_zeros(target.size(0), target.max() + 1)
    tmp.scatter_(1, target.view(-1, 1), 1.0)
    return tmp

def load_data(nb_data = 1000, flatten = False, one_hot_labels = False, normalize = False):
    mnist_train_set = datasets.MNIST(data_dir + '/mnist/', train=True, download=True)
    mnist_test_set = datasets.MNIST(data_dir + '/mnist/', train=False, download=True)

    train_input = mnist_train_set.data.view(-1, 1, 28, 28).float()[:nb_data]
    train_target = mnist_train_set.targets[:nb_data]
    test_input = mnist_test_set.data.view(-1, 1, 28, 28).float()[:nb_data]
    test_target = mnist_test_set.targets[:nb_data]

    if flatten:
        train_input = train_input.clone().reshape(train_input.size(0), -1)
        test_input = test_input.clone().reshape(test_input.size(0), -1)

    if one_hot_labels:
        train_target = convert_to_one_hot_labels(train_input, train_target)
        test_target = convert_to_one_hot_labels(test_input, test_target)

    if normalize:
        mu, std = train_input.mean(), train_input.std()
        train_input.sub_(mu).div_(std)
        test_input.sub_(mu).div_(std)

    return train_input, train_target, test_input, test_target

def plot_results(train_losses, train_errors, test_errors):

    epoch = len(train_losses)

    train_losses = torch.FloatTensor(train_losses)
    train_errors = torch.FloatTensor(train_errors)
    test_errors = torch.FloatTensor(test_errors)

    plt.subplot(121)
    plt.plot(range(epoch), train_losses)
    plt.legend(['Train loss'])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.subplot(122)
    plt.plot(range(epoch), train_errors*100)
    plt.plot(range(epoch), test_errors*100)
    plt.legend(['Train', 'Test'])
    plt.xlabel('Epoch')
    plt.ylabel('Error rate (in %)')
    plt.ylim(0, 100)

def plot_prediction(input, target, model):
    if target.ndim == 2: 
        # Convert one_hot to normal prediction
        target = target.argmax(dim=1)
    plt.figure()
    prediction = model(input)
    classes = torch.unique(target)
    _, predicted_classes = prediction.max(dim=1)
    for c in classes:
        plt.scatter(input[c == predicted_classes, 0],
                    input[c == predicted_classes, 1], label='class ' + str(c.item()))
    plt.scatter(input[predicted_classes != target, 0],
                input[predicted_classes != target, 1], color='red', label='misclassified')
    plt.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left",
               mode="expand", borderaxespad=0, ncol=3)
    plt.xlabel('x')
    plt.ylabel('y')

