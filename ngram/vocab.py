tokens = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,?!'’-+():\"\\“\\”\n"

char_to_idx = {ch: i for i, ch in enumerate(tokens)}
idx_to_char = {i: ch for i, ch in enumerate(tokens)}

encode = lambda s: [char_to_idx[c] for c in s]
decode = lambda l: ''.join([idx_to_char[i] for i in l])