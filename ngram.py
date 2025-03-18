import torch
import torch.nn as nn

class NGram(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.prob = nn.Embedding(vocab_size, vocab_size)

    def forward(self, x):
        return self.prob(x)

    def generate(self, x, size):
        result = x

        for _ in range(size):
            logits = self.forward(x)
            logits = logits[:, -1, :] # B x P
            prob = torch.softmax(logits, dim=-1)
            x = torch.multinomial(prob, num_samples=1) # B x 1

            result = torch.cat((result,x), dim=1)

        return result

    def forward_ngram(self, x, n=2):
        B, T = x.size()
        all_logits = []
        for t in range(T):
            start_idx = max(0, t - n + 1)
            context_tokens = x[:, start_idx: t + 1]
            logits_sum = 0
            for i in range(context_tokens.size(1)):
                logits_sum += self.prob(context_tokens[:, i])
            all_logits.append(logits_sum.unsqueeze(1))
        return torch.cat(all_logits, dim=1)

    def generate_ngram(self, x, size, n=2):
        result = x.clone()
        for _ in range(size):
            logits = self.forward_ngram(result, n)
            logits = logits[:, -1, :]
            probs = torch.softmax(logits, dim=-1)
            new_token = torch.multinomial(probs, num_samples=1)
            result = torch.cat((result, new_token), dim=1)
        return result


