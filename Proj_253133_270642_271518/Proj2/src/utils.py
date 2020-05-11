from torch import empty
import math
import torch 

def generate_set_crossEntropy(nb):
    input = empty(nb, 2).uniform_(0, 1)
    target = input.pow(2).sum(dim = 1).sub(1 / (2*math.pi)).sign().sub(-1).div(-2).long()
    return input, target

def generate_set_MSEloss(nb):
    input = empty(nb, 2).uniform_(0, 1)
    label = -input.pow(2).sum(dim = 1).sub(1 / (2*math.pi)).sign().sub(-1).div(-2).long()
    target = torch.zeros(nb, 2)
    for i in label: 
        target[i]=1
    return input, target

def generate_data(nb, normalize = False):
    train, train_target = generate_set_MSEloss(nb)
    test, test_target = generate_set_MSEloss(nb)

    if normalize:
        mu, std = train[0].mean(), train[0].std()
        train[0].sub_(mu).div_(std)
        test[0].sub_(mu).div_(std)

    return train, train_target, test, test_target


def plot_results(train_losses, train_errors, test_errors):
    import matplotlib.pyplot as plt

    epoch = len(train_losses[0])

    train_losses = torch.FloatTensor(train_losses)
    train_errors = torch.FloatTensor(train_errors)
    test_errors = torch.FloatTensor(test_errors)

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
    plt.legend(['Train', 'Test'])
    plt.xlabel('Epoch')
    plt.ylabel('Error rate (in %)')
    plt.show()
