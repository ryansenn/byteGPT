import torch
import torch.nn as nn

class NGram(nn.Module):
    def __init__(self, vocab_size):
        self.prob = nn.Embedding(vocab_size, vocab_size)

    def forward(self, x):
        return self.prob[x]

    def generate(self, x, size):

        for _ in range(size):
            logits = self.forward(x)
            logits = logits[:, -1, :]




