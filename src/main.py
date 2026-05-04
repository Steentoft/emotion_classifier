
from datasets import load_dataset

ds = load_dataset("dair-ai/emotion", "split")

test = ds["train"][0]

print(test)

print(test["text"].split())