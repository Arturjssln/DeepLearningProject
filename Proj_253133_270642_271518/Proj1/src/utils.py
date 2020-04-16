import torch
from torchvision import datasets

import os


######################################################################
# The data
data_dir = '../data'

def convert_to_one_hot_labels(input, target):
    tmp = input.new_zeros(target.size(0), target.max() + 1)
    tmp.scatter_(1, target.view(-1, 1), 1.0)
    return tmp

def load_data(cifar = None, one_hot_labels = False, normalize = False, flatten = True):
    if (cifar is not None and cifar):
        print('* Using CIFAR')
        cifar_train_set = datasets.CIFAR10(data_dir + '/cifar10/', train = True, download = True)
        cifar_test_set = datasets.CIFAR10(data_dir + '/cifar10/', train = False, download = True)

        train_input = torch.from_numpy(cifar_train_set.data)
        train_input = train_input.transpose(3, 1).transpose(2, 3).float()
        train_target = torch.tensor(cifar_train_set.targets, dtype = torch.int64)

        test_input = torch.from_numpy(cifar_test_set.data).float()
        test_input = test_input.transpose(3, 1).transpose(2, 3).float()
        test_target = torch.tensor(cifar_test_set.targets, dtype = torch.int64)

    else:
        print('* Using MNIST')
        mnist_train_set = datasets.MNIST(data_dir + '/mnist/', train = True, download = True)
        mnist_test_set = datasets.MNIST(data_dir + '/mnist/', train = False, download = True)

        train_input = mnist_train_set.data.view(-1, 1, 28, 28).float()
        train_target = mnist_train_set.targets
        test_input = mnist_test_set.data.view(-1, 1, 28, 28).float()
        test_target = mnist_test_set.targets

    if flatten:
        train_input = train_input.clone().reshape(train_input.size(0), -1)
        test_input = test_input.clone().reshape(test_input.size(0), -1)


    print('** Use {:d} train and {:d} test samples'.format(train_input.size(0), test_input.size(0)))

    if one_hot_labels:
        train_target = convert_to_one_hot_labels(train_input, train_target)
        test_target = convert_to_one_hot_labels(test_input, test_target)

    if normalize:
        mu, std = train_input.mean(), train_input.std()
        train_input.sub_(mu).div_(std)
        test_input.sub_(mu).div_(std)

    return train_input, train_target, test_input, test_target

######################################################################

def mnist_to_pairs(nb, input, target):
    input = torch.functional.F.avg_pool2d(input, kernel_size = 2)
    a = torch.randperm(input.size(0))
    a = a[:2 * nb].view(nb, 2)
    input = torch.cat((input[a[:, 0]], input[a[:, 1]]), 1)
    classes = target[a]
    target = (classes[:, 0] <= classes[:, 1]).long()
    return input, target, classes

######################################################################

def generate_pair_sets(nb, normalize = False):
    train_set = datasets.MNIST(data_dir + '/mnist/', train = True, download = True)
    train_input = train_set.data.view(-1, 1, 28, 28).float()
    train_target = train_set.targets

    test_set = datasets.MNIST(data_dir + '/mnist/', train = False, download = True)
    test_input = test_set.data.view(-1, 1, 28, 28).float()
    test_target = test_set.targets

    if normalize:
        mu, std = train_input.mean(), train_input.std()
        train_input.sub_(mu).div_(std)
        test_input.sub_(mu).div_(std)

    return mnist_to_pairs(nb, train_input, train_target) + \
           mnist_to_pairs(nb, test_input, test_target)

######################################################################

def generate_data(datasize, normalize):
    train_input, train_target, train_classes, \
    test_input, test_target, test_classes = generate_pair_sets(datasize, normalize)

    train_input = train_input.reshape(-1, 1, train_input.shape[-2], train_input.shape[-1])
    test_input = test_input.reshape(-1, 1, test_input.shape[-2], test_input.shape[-1])
    train_classes = train_classes.reshape(-1)
    test_classes = test_classes.reshape(-1)
    print("\n** Data generated **")
    return (train_input, train_target, train_classes, test_input, test_target, test_classes)

######################################################################


def plot_results(train_losses, train_errors, test_errors, goal_errors, force_error_axis = False, save=False, save_title=None):

    import matplotlib.pyplot as plt

    epoch = len(train_losses[0])

    train_losses = torch.FloatTensor(train_losses)
    train_errors = torch.FloatTensor(train_errors)
    test_errors = torch.FloatTensor(test_errors)
    goal_errors = torch.FloatTensor(goal_errors)

    plt.style.use('seaborn-whitegrid')
    plt.subplot(121)
    plt.errorbar(range(epoch), train_losses.mean(dim=0),
                 yerr=train_losses.std(dim=0), capsize=5, fmt='.', ls='--')
    plt.legend(['Train loss'])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.subplot(122)
    plt.errorbar(range(epoch), train_errors.mean(dim=0)*100,
                 yerr=train_errors.std(dim=0)*100, color='blue', capsize=5, fmt='.', ls='--')
    plt.errorbar(range(epoch), test_errors.mean(dim=0)*100,
                 yerr=test_errors.std(dim=0)*100, color='red', capsize=5, fmt='.', ls='--')
    plt.errorbar(range(epoch), goal_errors.mean(dim=0)*100,
                 yerr=goal_errors.std(dim=0)*100, color='green', capsize=5, fmt='.', ls='--')
    plt.legend(['Train', 'Test (predict digit)', 'Test (predict comparison)'])
    plt.xlabel('Epoch')
    plt.ylabel('Error rate (in %)')

    if force_error_axis:
        plt.ylim(0, 100)
    if save is True:
        plt.savefig('{}.png'.format(save_title))
    else:
        plt.show(block = False)

    

######################################################################

def count_parameters(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp
