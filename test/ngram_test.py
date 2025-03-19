from ngram.NGram import NGram
from data.vocab import *
import torch
import nltk
from nltk.corpus import treebank


nltk.download('treebank')
text = " ".join([" ".join(sent) for sent in treebank.sents()])


with open('../data/basic_data.txt', 'r', encoding='utf-8') as f:
    text += f.read()

with open('../data/data_sf.txt', 'r', encoding='utf-8') as f:
    text += f.read()


print(f"Total number of characters: {len(text)}")

train_data = torch.tensor(encode(text), dtype=torch.long)

def get_batch(batch_size):
    ix = torch.randint(len(train_data) - block_size, (batch_size,))
    x = torch.stack([train_data[i:i+block_size] for i in ix])
    y = torch.stack([train_data[i+block_size:i+block_size+1] for i in ix])
    return x, y

block_size = 16

model = NGram(block_size,len(tokens))

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

batch_size = 64
for steps in range(10000):
    xb, yb = get_batch(batch_size)
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print(loss.item())

x = torch.tensor([encode("hello my name is Ryan and today I am testing this model, ")])
res = model.generate(x,200)
print(decode(res[0].tolist()))