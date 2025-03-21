import torch
import torch.nn as nn


class Bigram(nn.Module):

    def __init__(self, vocab_size, n_embed=32, block_size=8):
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self, idx, targets=None):
        B, T = x.shape # batch x length

        tok_emb = self.token_embedding_table[idx] # batch x length x channels
        pos_emb = self.position_embedding_table[idx] # b x t x c

        x = tok_emb + pos_emb
        logits = self.lm_head(x) # b x t x v


B,T,C = 4,8,32
x = torch.randn(B, T, C)

head_size = 16
key = nn.Linear(C, head_size, bias=False)
query = nn.Linear(C, head_size, bias=False)

k = key(x) # b x t x 16
q = query(x) # b x t x 16
wei = q @ k.transpose(-2, -1) # (b,t,16) @ (b,16,t) = (b, t, t)
out = wei @ x

print(wei)




