import math
import torch 

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

def plot_results(train_losses, train_errors, test_errors):
    import matplotlib.pyplot as plt

    epoch = len(train_losses)

    train_losses = torch.FloatTensor(train_losses)
    train_errors = torch.FloatTensor(train_errors)
    test_errors = torch.FloatTensor(test_errors)

    plt.style.use('seaborn-whitegrid')
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
    plt.show()
