from ngram import NGram
from vocab import *
import torch
import torch.nn as nn

with open('data.txt', 'r', encoding='utf-8') as f:
    text = f.read()

data = torch.tensor(encode(text), dtype=torch.long)

n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

batch_size = 4 # how many independent sequences will we process in parallel?
block_size = 8 # what is the maximum context length for predictions?

def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

model = NGram(len(tokens))

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

xb, yb = get_batch('train')

batch_size = 32
for steps in range(10000):
    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print(loss.item())

x = torch.tensor([[1]])
res = model.generate(x,200)
print(decode(res[0].tolist()))