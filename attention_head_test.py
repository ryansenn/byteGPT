import torch
from self_attention import Head

h = Head(4, 1, 8)

x = torch.tensor([[[1],[2],[3],[4]], [[1],[2],[3],[4]]], dtype=torch.float32) # B x T x C = 2 x 4 x 1

out = h.forward(x)

print(out[0])