import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super().__init__()
        self.heads = heads
        self.head_dim = embed_size // heads
        self.query = nn.Linear(embed_size, embed_size, bias=False)
        self.key = nn.Linear(embed_size, embed_size, bias=False)
        self.value = nn.Linear(embed_size, embed_size, bias=False)
        self.fc_out = nn.Linear(embed_size, embed_size)

    def forward(self, x):
        B, T, C = x.shape  # Batch, Time, Channels
        H = self.heads
        d = self.head_dim

        q = self.query(x).view(B, T, H, d).transpose(1, 2)  # (B, H, T, d)
        k = self.key(x).view(B, T, H, d).transpose(1, 2)
        v = self.value(x).view(B, T, H, d).transpose(1, 2)

        attn_scores = (q @ k.transpose(-2, -1)) * (d ** -0.5)  # Scaled dot-product
        attn_weights = attn_scores.softmax(dim=-1)
        out = attn_weights @ v  # (B, H, T, d)

        out = out.transpose(1, 2).contiguous().view(B, T, C)  # Merge heads
        return self.fc_out(out)

# Example usage
x = torch.randn(2, 5, 16)  # (batch=2, seq_len=5, embed_dim=16)
attn = SelfAttention(embed_size=16, heads=4)
out = attn(x)
print(out.shape)  # (2, 5, 16)