import tiktoken
import torch
from datasets import load_dataset
import math
import torch.nn as nn
import torch.nn.functional as F

# Default encoder
encoder = tiktoken.get_encoding("o200k_base")

# own created encoder that uses the same as deafult but added 2 special tokens
ownEncoder = tiktoken.Encoding(
    name="o200k_base_own",
    pat_str=encoder._pat_str,
    mergeable_ranks=encoder._mergeable_ranks,
    special_tokens={
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
    return torch.tensor(ListPad, dtype=torch.long)


def createVocab(dataset):
    vocab = {"<PAD>": 0, "<UNK>": 1}
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
    convertetList1 = convertLength(list1, n)
    convertetList2 = convertLength(list2, n)
    convertetList3 = convertLength(list3, n)

    def longTensor(converted):
        return torch.tensor(converted, dtype=torch.long)

    return longTensor(convertetList1), longTensor(convertetList2), longTensor(convertetList3)

### MODEL 1 ###

class SelfAttention(nn.Module):
    def __init__(self, d_model, d_key):
        super().__init__()
        # Three separate linear layers for the queries, keys, and values
        self.w_q = nn.Linear(d_model, d_key)
        self.w_k = nn.Linear(d_model, d_key)
        self.w_v = nn.Linear(d_model, d_model)

    def forward(self, x):
        w_q = self.w_q(x)
        w_k = self.w_k(x)
        w_v = self.w_v(x)

        qk = w_q @ torch.transpose(w_k, -2, -1)

        qk = qk/math.sqrt(w_k.shape[-1])

        return torch.softmax(qk, 1) @ w_v


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, d_key, n_heads):
        super().__init__()
        self.heads = nn.ModuleList([SelfAttention(d_model, d_key) for _ in range(n_heads)])
        # Down projection back to model dimension
        # Alternatively, we could also split the input into n_heads and concatenate the output
        self.w_o = nn.Linear(n_heads * d_model, d_model)

    def forward(self, x):
        heads = []
        for head in self.heads:
            heads.append(head(x))

        return self.w_o(torch.cat(heads, -1))

class TransformerBlock(nn.Module):
    def __init__(self, d_model, d_key, n_heads, mlp_factor=4):
        super().__init__()
        # We need to init two layer norms because they have parameters
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, d_key, n_heads)
        self.ln2 = nn.LayerNorm(d_model)

        # a feedforward module with one internal hidden layer
        self.mlp = nn.Sequential(
            nn.Linear(d_model, mlp_factor * d_model),
            nn.SiLU(),  # Swish activation function, f(x) = x * sigmoid(x)
            nn.Linear(mlp_factor * d_model, d_model)
        )

    def forward(self, x):
        ln1 = self.ln1(x)

        attn = self.attn(ln1) + x

        x = self.ln2(attn)

        return self.mlp(x) + attn


class TransformerClassifier(nn.Module):
    def __init__(self, n_embeds, n_classes, d_model=256, d_key=64, n_heads=4, mlp_factor=4, n_layers=2):
        super().__init__()
        self.token_embedding = nn.Embedding(n_embeds, d_model)
        self.transformer_model = nn.Sequential(
            *[TransformerBlock(d_model, d_key, n_heads, mlp_factor) for _ in range(n_layers)])
        self.final_layer_norm = nn.LayerNorm(d_model)
        self.classifier = nn.Sequential(nn.Linear(d_model, d_model), nn.SiLU(), nn.Linear(d_model, n_classes))

    def forward(self, x):
        x = self.token_embedding(x)
        x = self.transformer_model(x)
        x = self.final_layer_norm(x)

        return self.classifier(torch.mean(x, -2))

def test_SelfAttention():
    # Test the Attention module
    att = SelfAttention(64, 16)
    x = torch.randn(32, 10, 64)
    z = att(x)
    assert ((z - x).abs() > 1e-6).any()
    assert z.shape == (32, 10, 64), z.shape

def test_MultiHeadAttention():
    # Test the MultiHeadAttention module
    mha = MultiHeadSelfAttention(64, 16, 8)
    x = torch.randn(32, 10, 64)
    z = mha(x)
    assert ((z - x).abs() > 1e-6).any()
    assert z.shape == (32, 10, 64), z.shape

def test_TransformerBlock():
    # Test the TransformerBlock module
    tb = TransformerBlock(64, 16, 8)
    x = torch.randn(32, 10, 64)
    z = tb(x)
    assert ((z - x).abs() > 1e-6).any()
    assert z.shape == (32, 10, 64), z.shape

def test_TransformerClassifier():
    # Test the TransformerClassifier module
    t = TransformerClassifier(2, 2)
    x = torch.randint(2, (32, 10))
    z = t(x)
    assert z.shape == (32, 2), z.shape

def main():
    dataset = load_dataset("dair-ai/emotion", "split")
    train, validation, test = encodeAllSplitsTiktoken(dataset)

    y_train = []
    for train_label in dataset["train"]:
        y_train.append(train_label["label"])

    y_val = []
    for val_label in dataset["train"]:
        y_val.append(val_label["label"])

    train, validation, test = convertLengthTiktoken(train, 30), convertLengthTiktoken(validation,30), convertLengthTiktoken(test,                                                                                                             30)
    trainTensor, validationTensor, testTensor = convert2Tensor(train), convert2Tensor(validation), convert2Tensor(test)

    model = TransformerClassifier(2, 2, d_model=8, d_key=8, n_heads=2, mlp_factor=4, n_layers=2) #97% 2426 parameters

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(500):
        model.train()
        optimizer.zero_grad()
        y_pred = model(trainTensor)
        loss = criterion(y_pred, y_train)
        loss.backward()
        optimizer.step()
        print(f'Epoch {epoch + 1}, Loss: {loss.item()}')
    # Check the validation accuracy
    with torch.no_grad():
        model.eval()
        y_pred = model(validationTensor)
        acc = (torch.argmax(y_pred, dim=1) == y_val).float().mean()
        print(f'Validation accuracy: {100 * acc.item()}%')

    # Number of parameters
    print(f'Number of parameters: {sum(p.numel() for p in model.parameters())}')



if __name__ == "__main__":
    main()

