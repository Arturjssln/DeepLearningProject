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
