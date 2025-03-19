import random

tokens = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,?!'’-+():\"\\“\”"

char_to_idx = {ch: i for i, ch in enumerate(tokens)}
idx_to_char = {i: ch for i, ch in enumerate(tokens)}

encode = lambda s: [char_to_idx.get(c, random.randint(0, len(tokens) - 1)) for c in s]
decode = lambda l: ''.join([idx_to_char.get(i, '?') for i in l])