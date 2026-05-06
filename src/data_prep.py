import tiktoken
import torch
from datasets import load_dataset


LABELS = ["sadness", "joy", "love", "anger", "fear", "surprise"]
MAX_LEN = 30
PAD_ID = 0
UNK_ID = 1


enc = tiktoken.get_encoding("o200k_base")


def pickDevice():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def encodeTextTiktoken(text):
    return enc.encode(text)


def encodeAllSplitsTiktoken(dataset):
    out = {"train": [], "validation": [], "test": []}
    for split in out:
        for row in dataset[split]:
            out[split].append(encodeTextTiktoken(row["text"]))
    return out["train"], out["validation"], out["test"]


def buildMapping(encodedTrain):
    mapping = {}
    nextId = 2
    for seq in encodedTrain:
        for tok in seq:
            if tok not in mapping:
                mapping[tok] = nextId
                nextId += 1
    return mapping


def applyRemap(encodedList, mapping):
    result = []
    for seq in encodedList:
        result.append([mapping.get(tok, UNK_ID) for tok in seq])
    return result


def convertLength(listLists, n):
    result = []
    for seq in listLists:
        if len(seq) >= n:
            result.append(seq[:n])
        else:
            result.append(list(seq) + [PAD_ID] * (n - len(seq)))
    return result


def convert2Tensor(padded):
    return torch.tensor(padded, dtype=torch.long)


def loadAndPrep(maxLength=MAX_LEN):
    dataset = load_dataset("dair-ai/emotion", "split")
    trainEnc, valEnc, testEnc = encodeAllSplitsTiktoken(dataset)

    mapping = buildMapping(trainEnc)
    vocabSize = len(mapping) + 2  # PAD + UNK

    trainRemap = applyRemap(trainEnc, mapping)
    valRemap = applyRemap(valEnc, mapping)
    testRemap = applyRemap(testEnc, mapping)

    trainPad = convertLength(trainRemap, maxLength)
    valPad = convertLength(valRemap, maxLength)
    testPad = convertLength(testRemap, maxLength)

    trainX = convert2Tensor(trainPad)
    valX = convert2Tensor(valPad)
    testX = convert2Tensor(testPad)

    trainY = torch.tensor([row["label"] for row in dataset["train"]], dtype=torch.long)
    valY = torch.tensor([row["label"] for row in dataset["validation"]], dtype=torch.long)
    testY = torch.tensor([row["label"] for row in dataset["test"]], dtype=torch.long)

    return {
        "trainX": trainX,
        "valX": valX,
        "testX": testX,
        "trainY": trainY,
        "valY": valY,
        "testY": testY,
        "mapping": mapping,
        "vocabSize": vocabSize,
    }
