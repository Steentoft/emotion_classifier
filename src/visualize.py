import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from datasets import load_dataset

from data_prep import enc

def main():
    ds = load_dataset("dair-ai/emotion", "split")
    train = ds["train"]

    lengths = [len(enc.encode(row["text"])) for row in train]

    print(f"median: {np.median(lengths)}")
    print(f"std:    {np.std(lengths):.2f}")

    c = Counter(lengths)
    items = sorted(c.most_common())
    x = [k for k, _ in items]
    y = [v for _, v in items]

    fig, ax = plt.subplots()
    ax.bar(x, y)
    ax.set_xlabel("Tokens per sentence (Encoder)")
    ax.set_ylabel("Count")
    ax.set_title("Train split token-length distribution")
    plt.show()
