import torch
import torch.nn as nn

class NGram(nn.Module):
    def __init__(self, n, vocab_size, embed_size=8):
        super().__init__()
        self.n = n
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.fc = nn.Linear(n*embed_size, vocab_size)

    def forward(self, x, targets=None):
        emb = self.embed(x).view(x.shape[0], -1)
        logits = self.fc(emb)

        if targets is None:
            return logits
        else:
            targets = targets.view(-1)
            loss = nn.functional.cross_entropy(logits, targets)

        return logits, loss


    def generate(self, x, size):
        result = x

        for _ in range(size):
            logits = self.forward(result[:, -self.n:])
            prob = torch.softmax(logits, dim=-1)
            n_x = torch.multinomial(prob, num_samples=1)
            result = torch.cat((result, n_x), dim=1)

        return result

