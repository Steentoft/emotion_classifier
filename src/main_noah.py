from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from datasets import load_dataset
import tiktoken
from collections import Counter

enc = tiktoken.get_encoding("o200k_base")

ds = load_dataset("dair-ai/emotion", "split")
dns = load_dataset("dair-ai/emotion", "unsplit")

test = ds["test"]
train = ds["train"]
validation = ds["validation"]



def createVocab(split):
    vocab = {"PAD" : 0, "Unknown" : 1}
    listText = []
    count = 2
    for text in split["train"]:
        for word in text["text"].split():
            listText.append(word) 
    
    for word in listText:
        vocab.update({word : count})
        count += 1

    return vocab


def encodeText(wordList, vocab):
    for i in range(len(wordList)):
        if wordList[i] in vocab.keys():
            wordList[i] = vocab[wordList[i]]
        else:
            wordList[i] = 1

def encodeAllSplits(dataset):
    fullVocabulary = createVocab(dns)
    trainSplit = []
    validationSplit = []
    testSplit = []
    for split in dataset:
        for text in ds[split]:
            temp = []
            for word in text["text"].split():
                temp.append(word)
            if split == "train":
                    trainSplit.append(temp)
            elif split == "validation":
                    validationSplit.append(temp)
            elif split == "test":
                    testSplit.append(temp)

    for wordList in trainSplit:
        encodeText(wordList, fullVocabulary)
    for wordList in validationSplit:
        encodeText(wordList, fullVocabulary)
    for wordList in testSplit:
        encodeText(wordList, fullVocabulary)

    return trainSplit, validationSplit, testSplit 
        
            
encodeAllSplits(ds)
exit()
def tokenize_list(split):
    result = []
    for text in split:
        for word in text["text"].split():
            result.append(word)
    result = set(result)
    return list(result)

tl = tokenize_list(test)


vocabulary = {"PAD": 0, "Unknown": 1}
counter = 2
for word in tl:
    vocabulary.update({word : counter})
    counter += 1


print(vocabulary)

rev_multidict = {}
for key, value in vocabulary.items():
    rev_multidict.setdefault(value, set()).add(key)

print("test")
print([key for key, value in rev_multidict.items() if len(value) > 1])
print(len(rev_multidict))
print(len(vocabulary))

