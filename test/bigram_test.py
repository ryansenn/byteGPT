from ngram.bigram import Bigram
from data.vocab import *
import torch

with open('../data/data_sf.txt', 'r', encoding='utf-8') as f:
    text = f.read()

with open('../data/basic_data.txt', 'r', encoding='utf-8') as f:
    text += f.read()

train_data = torch.tensor(encode(text), dtype=torch.long)

batch_size = 4 # how many independent sequences will we process in parallel?
block_size = 8 # what is the maximum context length for predictions?

def get_batch(split):
    ix = torch.randint(len(train_data) - block_size, (batch_size,))
    x = torch.stack([train_data[i:i+block_size] for i in ix])
    y = torch.stack([train_data[i+1:i+block_size+1] for i in ix])
    return x, y

model = Bigram(len(tokens))

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