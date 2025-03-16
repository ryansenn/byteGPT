import torch
import torch.nn as nn

t = torch.Tensor([[[1.0,2.0,3.0],[4.0,5.0,6.0],[7.0,8.0,9.0]],[[1.0,2.0,3.0],[4.0,5.0,6.0],[7.0,8.0,9.0]]])
s = torch.softmax(t, dim=2)
print(t)
print(s)