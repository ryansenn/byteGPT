from ngram import NGram
from vocab import *
import torch
import torch.nn as nn

model = NGram(len(tokens))
x = torch.tensor([[1]])
res = model.generate(x,20)
print(decode(res[0].tolist()))