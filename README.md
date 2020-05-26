# DeepLearningProjects

This repository contains 2 projects:
- **Project 1**:  
It implements differents Neural Networks architecutre to compare their accuracy on MNIST dataset.

- **Project 2**:  
It implements a basic Neural Network framework that can be used similarly to the one implemented in Pytorch.

# Project 1 – Classification, weight sharing, auxiliary losses

## Requirements
torch  
torchvision  

For Linux and Windows, run the following command:
```
conda install pytorch torchvision cudatoolkit=10.2 -c pytorch
```
For MacOS, run the following command:
```
conda install pytorch torchvision -c pytorch
```

## Usage
Run the following command from the src directory:
```bash
python test.py

Options:
    --datasize INT              Size of dataset (same for train and test)
                                (defaut: 1000)
    --architecture STR          Choice of architecture
                                    (Possibility: 'linear', 'lenet', 'resnet', 'alexnet'; default: 'resnet')
    --nodes INT                 Number of nodes 
                                    (ignored if architecture is not 'linear'; default: 32)
    --epoch INT                 Number of epoch 
                                    (default: 25)
    --optimizer OPTI            Define optimizer to use
                                    (Possibility: 'MSE' or 'Adam'; default: None)
    --nb_residual_blocks INT    Set number of residual blocks
                                    (ignored if architecture is not 'resnet'; default: 2)
    --nb_channels INT           Set number of channels
                                    (ignored if architecture is not 'resnet'; default: 16)
    --kernel_size INT           Set kernel size
                                    (ignored if architecture is not 'resnet'; default: 7)
    --residual                  Use residual Neural Network
    --bn                        Use batch normalization
    --dropout                   Use dropout
    --auxloss                   Use auxiliary loss
```

# Project 2 – Neural network framework

## Requirements
torch  
torchvision  

For Linux and Windows, run the following command:
```
conda install pytorch torchvision cudatoolkit=10.2 -c pytorch
```
For MacOS, run the following command:
```
conda install pytorch torchvision -c pytorch
```

## Usage
### Framework usage
```python
import torch
import framework as ff

class Net(ff.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layer = ff.Sequential(
            ff.Linear(10, 200), 
            ff.ReLU(), 
            ff.Linear(200, 10))

    def forward(self, x):
        x = self.layer(x)
        return x

    def backward(self, criterion):
        d = criterion.backward()
        d = self.layer.backward(d)
        return d

    def train_(self, data, target, \
            criterion=ff.MSELoss()):
        pred = self(data)
        loss = criterion(pred, target)
        self.zero_grad()
        self.backward(criterion)
        # Do your training stuff here
```
### Run example
To run our **linear model**, run the following command from the src directory:
```bash
python test.py

Options:
    --loss LOSS     Loss to use.
                    (available: 'CrossEntropy' or 'MSE'; default: MSE)
    --notrain      Load best stored model and display its prediction on the test_data.
```

To run our **convolutional model**, run the following command from the src directory:
```bash
python test_conv.py

Options:
    --train         Train model instead of loading the saved weights of the model.
```
By default this script loads the best stored model and display its prediction on the test data.
Indeed, the training of this model can long given that convolutional layers are not optimised.  
Instead of using the initial dataset, this model is using the MNIST dataset.  
This model is very basic and use the following layers:  
`Cond2d, BatchNorm2d, ReLU, MaxPool2d, Dropout`, it is then followed by 2 fully connected linear layers.
