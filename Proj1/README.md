# Project 1 – Classification, weight sharing, auxiliary losses

## Requirements
torch  
torchvision  

### If you have a CUDA capable GPU 
For Linux and Windows, run the following command:
```
conda install pytorch torchvision cudatoolkit=10.2 -c pytorch
```

For MacOS, run the following command (doesn't support CUDA):
```
conda install pytorch torchvision -c pytorch
```

### If you want to run on CPU only
For Linux and Windows, run the following command:
```
conda install pytorch torchvision cpuonly -c pytorch
```

For MacOS, run the following command (doesn't support CUDA):
```
conda install pytorch torchvision -c pytorch
```

For any other installation (through PIP of LibTorch) please follow this link: https://pytorch.org

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