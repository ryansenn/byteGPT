import torch
import nltk
from nltk.corpus import treebank
from bigram_self_attention import Bigram
from data.vocab import encode, decode, tokens

nltk.download('treebank')
text = " ".join(" ".join(sent) for sent in treebank.sents())

with open('data/basic_data.txt', 'r', encoding='utf-8') as f:
    text += f.read()
with open('data/data_sf.txt', 'r', encoding='utf-8') as f:
    text += f.read()

train_data = torch.tensor(encode(text), dtype=torch.long)
block_size = 16

def get_batch(batch_size=64):
    ix = torch.randint(len(train_data) - block_size, (batch_size,))
    x = torch.stack([train_data[i:i+block_size] for i in ix])
    y = torch.stack([train_data[i+1:i+block_size+1] for i in ix])
    return x, y

model = Bigram(len(tokens), block_size=block_size)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

for step in range(10000):
    xb, yb = get_batch()
    logits, loss = model(xb, yb)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 500 == 0:
        print(f"Step {step}: loss = {loss.item():.4f}")

prompt = torch.tensor([encode("hello my name is Ryan and today I am testing this model, ")])
generated = model.generate(prompt, 200)
print(decode(generated[0].tolist()))
