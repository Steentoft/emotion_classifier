import numpy
from datasets import load_dataset
import tiktoken

enc = tiktoken.get_encoding("o200k_base")

ds = load_dataset("dair-ai/emotion", "split")

test = ds["train"]

def tokenize_split(split):
    result = []
    for text in split:
        result.append({ "text" : enc.encode(text["text"]), "label" : text["label"] })
    return result

list = tokenize_split(test)

tokens = []
for i in range(len(list)):
    tokens.append(len(list[i]["text"]))

print(tokens)

print(numpy.median(tokens))


dict = {}
for text in test:
    if len(text["text"]) in dict:
        dict.append({ len(text["text"]) : 1 })
    else:
        dict[len(text["text"])] = dict[len(text["text"])] + 1
print(dict)