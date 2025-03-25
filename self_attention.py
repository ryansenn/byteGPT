import torch
import torch.nn as nn
class Head(nn.Module):

    def __init__(self, block_size, n_embd, head_size):
        super().__init__()
        self.head_size = head_size
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.tril = torch.tril(torch.ones(block_size, block_size))

    def forward(self, x):
        B, T, C = x.shape

        q = self.query(x) # B, T, head_size
        k = self.key(x) # B, T, head_size

        wei = q @ k.transpose(-2,-1) * C**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = nn.functional.softmax(wei, dim=-1)

        v = self.value(x)

        out = wei @ v

        return out





