from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from datasets import load_dataset
import tiktoken
from collections import Counter

enc = tiktoken.get_encoding("o200k_base")

ds = load_dataset("dair-ai/emotion", "split")

test = ds["train"]

def tokenize_list(split):
    result = []
    for text in split:
        for word in text["text"].split():
            result.append(word)
    result = set(result)
    return list(result)

tl = tokenize_list(test)

vocabulary = {}
counter = 1
for word in tl:
    vocabulary.update({word : counter})
    counter += 1
    print(counter)

vocabulary.update({"PAD" : 0})

print(vocabulary)

rev_multidict = {}
for key, value in vocabulary.items():
    rev_multidict.setdefault(value, set()).add(key)

print("test")
print([key for key, value in rev_multidict.items() if len(value) > 1])
print(len(rev_multidict))
print(len(vocabulary))

