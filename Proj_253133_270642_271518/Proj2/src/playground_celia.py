import torch

drop = (torch.empty((3, 5, 6, 6)).uniform_() > 0.99).type(torch.IntTensor)
print(drop)