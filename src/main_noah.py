import tiktoken
import torch
from datasets import load_dataset

# Default encoder
encoder = tiktoken.get_encoding("o200k_base")

# own created encoder that uses the same as deafult but added 2 special tokens
ownEncoder = tiktoken.Encoding(
        name = "o200k_base_own",
        pat_str= encoder._pat_str,
        mergeable_ranks= encoder._mergeable_ranks,
        special_tokens= {
            **encoder._special_tokens,
            "<|pad|>": 200019,
            "<|unk|>": 200020,
            },
        )

def encodeTextTiktoken(text):
    """ Returns tiktoken o200k_base_own text """
    return ownEncoder.encode(text)

def encodeAllSplitsTiktoken(dataset):
    """ Takes a splitted dataset and returns a list of lists of ints for each split """
    train = []
    validation = []
    test = []
    for split in dataset:
        for text in dataset[split]:
            if split == "train":
                train.append(encodeTextTiktoken(text["text"]))
            elif split == "validation":
                validation.append(encodeTextTiktoken(text["text"]))
            elif split == "test":
                test.append(encodeTextTiktoken(text["text"]))

    return train, validation, test

def convertLengthTiktoken(encodedList, maxLength):
    """ Takes a encodedList and a maxLenght, and returns the encodedList with list that are either truncated or with padding """
    result = []
    for sequence in encodedList:
        if len(sequence) == maxLength:
            result.append(sequence)
        elif len(sequence) > maxLength:
            result.append(sequence[0:maxLength])
        else:
            result.append(sequence + [200019] * (maxLength - len(sequence)))
    return result

def convert2Tensor(ListPad):
    """ takes a list with lists of same length and return the tensor """
    return torch.tensor(ListPad, dtype = torch.long)



def createVocab(dataset):
    vocab = {"<PAD>" : 0, "<UNK>" : 1}
    listText = []
    count = 2
    for split in dataset["train"]:
        for word in split["text"].split():
            listText.append(word) 
    
    for word in listText:
        if word not in vocab:
            vocab[word] = count
            count += 1 

    return vocab

def encodeText(wordList, vocab):
    for i in range(len(wordList)):
        if wordList[i] in vocab.keys():
            wordList[i] = vocab[wordList[i]]
        else:
            wordList[i] = 1

def encodeAllSplits(dataset):
    fullVocabulary = createVocab(dataset)
    trainSplit = []
    validationSplit = []
    testSplit = []
    for split in dataset:
        for text in dataset[split]:
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
        


# tensor length = men + std (rounded up), 18 + 11.3 = 30

def convertLength(listLists, n):
    result = []
    for lists in listLists:
        newList = []
        if len(lists) == n: 
            newList = lists
            result.append(newList)
        elif len(lists) > n:
            newList = lists[0:n]
            result.append(newList)
        else:
            add = n - len(lists)
            newList += lists + add * [0]
            result.append(newList)
    return result

def createTensorFromMultipleLists(list1, list2, list3, n):
    convertetList1 = convertLength(list1,n)
    convertetList2 = convertLength(list2,n)
    convertetList3 = convertLength(list3,n)

    def longTensor(converted):
        return torch.tensor(converted, dtype = torch.long)
    return longTensor(convertetList1), longTensor(convertetList2), longTensor(convertetList3)


def main():
    dataset = load_dataset("dair-ai/emotion", "split")
    train, validation, test = encodeAllSplitsTiktoken(dataset)
    train, validation, test = convertLengthTiktoken(train, 30), convertLengthTiktoken(validation, 30), convertLengthTiktoken(test, 30)
    trainTensor, validationTensor, testTensor = convert2Tensor(train), convert2Tensor(validation), convert2Tensor(test)
    print("train")
    print(trainTensor)

    print("\n")
    print("validation")
    print(validationTensor)
    
    print("\n")
    print("test")
    print(testTensor)
    
    print("\n")
    # Shape should be [16000, 30]
    print(trainTensor.shape)
    # No unknown words in train tensor, so should return 0 
    print((trainTensor == 200020).sum())
    
    print((validationTensor == 200020).sum())


if __name__ == "__main__": 
    main()
    
