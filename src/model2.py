import torch
import torch.nn as nn
import torch.optim as optim

from data_prep import loadAndPrep, pickDevice, PAD_ID


def masked_mean_pool(embeddings: torch.Tensor,
                     attention_mask: torch.Tensor) -> torch.Tensor:
    mask_expanded = attention_mask.unsqueeze(-1).float()
    masked_emb = embeddings * mask_expanded
    sum_emb = masked_emb.sum(dim=1)
    token_counts = attention_mask.sum(dim=1, keepdim=True).float()
    return sum_emb / token_counts.clamp(min=1e-9)


class TextMLP(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes,
                 pad_idx=PAD_ID, dropout=0.3):
        super().__init__()
        self.pad_idx = pad_idx
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, input_ids):
        emb = self.embedding(input_ids)
        mask = (input_ids != self.pad_idx)
        pooled = masked_mean_pool(emb, mask)
        x = self.fc1(pooled)
        x = self.relu(x)
        x = self.dropout(x)
        return self.fc2(x)


class TextRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes,
                 pad_idx=PAD_ID):
        super().__init__()
        self.pad_idx = pad_idx
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, input_ids):
        emb = self.embedding(input_ids)
        lengths = (input_ids != self.pad_idx).sum(dim=1).clamp(min=1)
        packed = nn.utils.rnn.pack_padded_sequence(
            emb, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, hidden = self.rnn(packed)
        hidden = hidden.squeeze(0)
        return self.fc(hidden)


def main(lr=1e-3, embed_dim=64, hidden_dim=256, epochs=50):
    data = loadAndPrep()
    device = pickDevice()
    print(f"Using device: {device}")
    print(f"Vocab size: {data['vocabSize']}")

    model = TextRNN(data["vocabSize"], embed_dim, hidden_dim, num_classes=6).to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_state = None

    for epoch in range(epochs):
        model.train()
        for batch, targets in data["trainLoader"]:
            batch = batch.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            opt.zero_grad()
            logits = model(batch)
            loss = loss_fn(logits, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        model.eval()
        correct = 0
        total = 0
        total_loss = 0.0
        with torch.no_grad():
            for batch, targets in data["validationLoader"]:
                batch = batch.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                logits = model(batch)
                loss = loss_fn(logits, targets)
                total_loss += loss.item() * targets.size(0)
                preds = logits.argmax(dim=1)
                total += targets.size(0)
                correct += (preds == targets).sum().item()

        val_acc = 100 * correct / total
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        print(f"Epoch {epoch+1:3d} | val loss {total_loss/total:.4f} | "
              f"val acc {val_acc:.2f} | best {best_val_acc:.2f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"Best val acc: {best_val_acc:.2f}%")


if __name__ == "__main__":
    main()
