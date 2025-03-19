import torch
import torch.nn as nn

class Bigram(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.prob = nn.Embedding(vocab_size, vocab_size)

    def forward(self, x, targets=None):
        logits = self.prob(x)  # (B,T,C)

        if targets is None:
            return logits
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = nn.functional.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, x, size):
        result = x

        for _ in range(size):
            logits = self.forward(x)
            logits = logits[:, -1, :] # B x P
            prob = torch.softmax(logits, dim=-1)
            x = torch.multinomial(prob, num_samples=1) # B x 1

            result = torch.cat((result,x), dim=1)

        return result



