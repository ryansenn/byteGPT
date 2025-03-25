import torch
import torch.nn as nn
from self_attention import Head


class Bigram(nn.Module):
    def __init__(self, vocab_size, n_embed=32, block_size=8):
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.sa_head = Head(block_size, n_embed, n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        tok_emb = self.token_embedding_table[idx]
        pos_emb = self.position_embedding_table[idx]

        x = tok_emb + pos_emb
        x = self.sa_head(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = nn.functional.cross_entropy(logits, targets)

        return logits, loss




