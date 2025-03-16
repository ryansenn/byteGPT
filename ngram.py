import torch
import torch.nn as nn

class NGram(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.prob = nn.Embedding(vocab_size, vocab_size)

    def forward(self, x):
        return self.prob(x)

    def generate(self, x, size):
        result = [x]

        for _ in range(size):
            logits = self.forward(x)
            logits = logits[:, -1, :] # B x P
            prob = torch.softmax(logits, dim=-1)
            x = torch.multinomial(prob, num_samples=1) # B x 1

            result.append(x)

        return torch.cat(result)



model = NGram(10)
x = torch.tensor([[1]])
res = model.generate(x,20)
print(res)


