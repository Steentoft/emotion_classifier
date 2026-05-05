import matplotlib.pyplot as plt
import numpy as np
from datasets import load_dataset
import tiktoken
from collections import Counter

enc = tiktoken.get_encoding("o200k_base")

ds = load_dataset("dair-ai/emotion", "split")

test = ds["train"]

     

def tokenize_split(split):
    result = []
    for text in split:
        result.append({ "text" : [enc.decode_single_token_bytes(token) for token in enc.encode(text["text"])], "label" : text["label"] })
    return result

list = tokenize_split(test)
print(list)

tokens = []
for t in list:
    tokens.append(len(t["text"]))


print(np.median(tokens))
print(np.std(tokens))

c = Counter(tokens)

print(sorted(c.most_common()))

x_val = [x[0] for x in sorted(c.most_common())]
y_val = [x[1] for x in sorted(c.most_common())]

# plot:
fig, ax = plt.subplots()

ax.bar(x_val, y_val)

plt.show()
