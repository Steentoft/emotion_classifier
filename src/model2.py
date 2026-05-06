import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from datasets import load_dataset

# ── Dataset ──────────────────────────────────────────────────────────────────

dataset = load_dataset("dair-ai/emotion", "split")

train_sentences = []
train_labels = []
for data in dataset["train"]:
    train_sentences.append(data["text"])
    train_labels.append(data["label"])

val_sentences = []
val_labels = []
for data in dataset["validation"]:
    val_sentences.append(data["text"])
    val_labels.append(data["label"])

#sentences = [
#    "the film was wonderful and touching",
#    "terrible movie very boring and slow",
#    "great performances and a beautiful story",
#    "awful acting the plot made no sense",
#    "loved every moment of this masterpiece",
#    "waste of time poorly written script",
#]
#labels = [1, 0, 1, 0, 1, 0]

def whitespace_tokenize(text: str) -> list[str]:
    return text.lower().split()

# Verification
print(whitespace_tokenize("The film was GREAT"))
# ['the', 'film', 'was', 'great']

def char_tokenize(text: str) -> list[str]:
    return [ch for ch in text.lower() if ch != " "]

# Verification
print(char_tokenize("Hi NLP"))
# ['h', 'i', 'n', 'l', 'p']

class Vocabulary:
    PAD_TOKEN = "<PAD>"   # index 0
    UNK_TOKEN = "<UNK>"   # index 1

    def __init__(self):
        self.token2idx: dict[str, int] = {}
        self.idx2token: dict[int, str] = {}

    def build(self, tokenized_corpus: list[list[str]]) -> None:
        # Reserve indices 0 and 1 for special tokens
        for special in (self.PAD_TOKEN, self.UNK_TOKEN):
            idx = len(self.token2idx)
            self.token2idx[special] = idx
            self.idx2token[idx] = special

        for sentence in tokenized_corpus:
            for token in sentence:
                if token not in self.token2idx:
                    idx = len(self.token2idx)
                    self.token2idx[token] = idx
                    self.idx2token[idx] = token

    def encode(self, tokens: list[str]) -> list[int]:
        unk_idx = self.token2idx[self.UNK_TOKEN]
        return [self.token2idx.get(t, unk_idx) for t in tokens]

    def decode(self, indices: list[int]) -> list[str]:
        return [self.idx2token[i] for i in indices]

    def __len__(self) -> int:
        return len(self.token2idx)


# Build and inspect
tokenized = [whitespace_tokenize(s) for s in train_sentences]
vocab = Vocabulary()
vocab.build(tokenized)

print(f"Vocabulary size: {len(vocab)}")          # 2 special + unique words
print(f"<PAD> index: {vocab.token2idx['<PAD>']}")  # 0
print(f"<UNK> index: {vocab.token2idx['<UNK>']}")  # 1

encoded = vocab.encode(tokenized[0])
print(f"Encoded: {encoded}")
print(f"Decoded: {vocab.decode(encoded)}")

def pad_sequences(sequences: list[list[int]], pad_idx: int = 0) -> torch.Tensor:
    max_len = max(len(s) for s in sequences)
    padded = [s + [pad_idx] * (max_len - len(s)) for s in sequences]
    return torch.tensor(padded, dtype=torch.long)


# Build the full padded batch
encoded_corpus = [vocab.encode(tok) for tok in tokenized]
padded_batch = pad_sequences(encoded_corpus)
print(padded_batch.shape)   # torch.Size([6, 6])
print(padded_batch)

# --- Validation set preparation ---
val_tokenized = [whitespace_tokenize(s) for s in val_sentences]
val_encoded = [vocab.encode(tok) for tok in val_tokenized]
val_padded = pad_sequences(val_encoded, pad_idx=vocab.token2idx["<PAD>"])
val_targets = torch.tensor(val_labels)
val_lengths = (val_padded != vocab.token2idx["<PAD>"]).sum(dim=1)


def build_embedding_layer(vocab_size: int,
                           embed_dim: int,
                           pad_idx: int = 0) -> nn.Embedding:
    emb = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
    # padding_idx ensures:
    #   (a) the <PAD> vector is initialised to zeros
    #   (b) its gradient is zeroed out so it is never updated
    return emb


EMBED_DIM = 16
embedding_layer = build_embedding_layer(len(vocab), EMBED_DIM)

# Sanity check: <PAD> embedding is all zeros
print(embedding_layer(torch.tensor([0])))   # tensor([[0., 0., …, 0.]])

# padded_batch: (6, 6)  →  embedded: (6, 6, 16)
embedded = embedding_layer(padded_batch)
print(embedded.shape)   # torch.Size([6, 6, 16])

# Confirm dimensions match expectations
B, T, D = embedded.shape

def masked_mean_pool(embeddings: torch.Tensor,
                     attention_mask: torch.Tensor) -> torch.Tensor:
    """
    embeddings     : (B, T, D)  — token embeddings
    attention_mask : (B, T)     — True for real tokens, False for <PAD>
    returns        : (B, D)     — sentence-level mean embedding
    """
    # Expand mask to (B, T, D) so it broadcasts over the embedding dimension
    mask_expanded = attention_mask.unsqueeze(-1).float()          # (B, T, 1)

    # Zero out padding positions
    masked_emb = embeddings * mask_expanded                        # (B, T, D)

    # Sum over T, then divide by the count of real tokens per row
    sum_emb   = masked_emb.sum(dim=1)                             # (B, D)
    token_counts = attention_mask.sum(dim=1, keepdim=True).float()# (B, 1)

    # Clamp to avoid division by zero for hypothetical all-padding rows
    return sum_emb / token_counts.clamp(min=1e-9)                 # (B, D)


# Quick test
attention_mask = (padded_batch != 0)   # True where token is not <PAD>
pooled = masked_mean_pool(embedded, attention_mask)
print(pooled.shape)   # torch.Size([6, 16])

class TextMLP(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 embed_dim: int,
                 hidden_dim: int,
                 num_classes: int,
                 pad_idx: int = 0,
                 dropout: float = 0.3):
        super().__init__()
        self.pad_idx  = pad_idx
        self.embedding = nn.Embedding(vocab_size, embed_dim,
                                      padding_idx=pad_idx)
        self.fc1      = nn.Linear(embed_dim, hidden_dim)
        self.relu     = nn.ReLU()
        self.dropout  = nn.Dropout(dropout)
        self.fc2      = nn.Linear(hidden_dim, num_classes)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # 1. Embedding lookup         (B, T) → (B, T, D)
        emb = self.embedding(input_ids)

        # 2. Build attention mask     (B, T) — True for real tokens
        mask = (input_ids != self.pad_idx)

        # 3. Masked mean pooling      (B, T, D) → (B, D)
        pooled = masked_mean_pool(emb, mask)

        # 4. MLP
        x = self.fc1(pooled)   # (B, D) → (B, H)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)        # (B, H) → (B, num_classes)
        return x               # raw logits; use nn.CrossEntropyLoss outside


# Smoke test
mlp = TextMLP(vocab_size=len(vocab), embed_dim=EMBED_DIM,
              hidden_dim=32, num_classes=6)
logits = mlp(padded_batch)
print(logits.shape)   # torch.Size([6, 2])

class TextRNN(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 embed_dim: int,
                 hidden_dim: int,
                 num_classes: int,
                 pad_idx: int = 0):
        super().__init__()
        self.pad_idx   = pad_idx
        self.embedding = nn.Embedding(vocab_size, embed_dim,
                                      padding_idx=pad_idx)
        self.rnn       = nn.RNN(embed_dim, hidden_dim,
                                batch_first=True)
        self.fc        = nn.Linear(hidden_dim, num_classes)

    def forward(self,
                input_ids: torch.Tensor,
                lengths: torch.Tensor) -> torch.Tensor:
        """
        input_ids : (B, T) — padded token indices, batch sorted by length desc
        lengths   : (B,)   — actual (non-padding) length of each sequence
        returns   : (B, num_classes)
        """
        # 1. Embedding lookup   (B, T) → (B, T, D)
        emb = self.embedding(input_ids)

        # 2. Pack padded sequence — tells the RNN to skip padding positions
        packed = pack_padded_sequence(emb, lengths.cpu(),
                                      batch_first=True,
                                      enforce_sorted=False)

        # 3. RNN forward pass
        _, hidden = self.rnn(packed)
        # hidden shape: (num_layers * num_directions, B, H) = (1, B, H)

        # 4. Squeeze out the layer dimension → (B, H)
        hidden = hidden.squeeze(0)

        # 5. Classification head  (B, H) → (B, num_classes)
        return self.fc(hidden)


# Compute real lengths (count of non-padding tokens per row)
lengths = (padded_batch != 0).sum(dim=1)
print(f"Sequence lengths: {lengths.tolist()}")

rnn = TextRNN(vocab_size=len(vocab), embed_dim=EMBED_DIM,
              hidden_dim=32, num_classes=6)
logits_rnn = rnn(padded_batch, lengths)
print(logits_rnn.shape)   # torch.Size([6, 2])

# Quick training loop (no validation, illustrative only)
# Quick training loop with validation
import torch.optim as optim

model = TextRNN(len(vocab), EMBED_DIM, 32, 6)
opt = optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

targets = torch.tensor(train_labels)
lengths = (padded_batch != 0).sum(dim=1)

best_val_acc = 0.0
best_state = None
per_epoch = 10

for epoch in range(500):
    model.train()
    opt.zero_grad()
    logits = model(padded_batch, lengths)
    loss = loss_fn(logits, targets)
    loss.backward()
    opt.step()

    if (epoch + 1) % per_epoch == 0:
        model.eval()
        with torch.no_grad():
            val_logits = model(val_padded, val_lengths)
            val_loss = loss_fn(val_logits, val_targets)
            val_preds = val_logits.argmax(dim=1)
            val_acc = (val_preds == val_targets).float().mean().item()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }

        print(
            f"Epoch {epoch+1:3d} | "
            f"train loss {loss.item():.4f} | "
            f"val loss {val_loss.item():.4f} | "
            f"val acc {val_acc:.2f} | "
            f"best val acc {best_val_acc:.2f}"
        )

if best_state is not None:
    model.load_state_dict(best_state)
