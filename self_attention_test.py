import torch
from self_attention import Head
from bigram_self_attention import Bigram
from data.vocab import *

'''
h = Head(4, 1, 8)
x = torch.tensor([[[1],[2],[3],[4]], [[1],[2],[3],[4]]], dtype=torch.float32) # B x T x C = 2 x 4 x 1
out = h.forward(x)
print(out[0])
'''

b = Bigram(len(tokens))
