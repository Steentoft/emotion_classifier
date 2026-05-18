import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score

from data_prep import loadAndPrep, PAD_ID, UNK_ID


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


class TextBiGRU(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes,
                 pad_idx=PAD_ID, dropout=0.3, num_layers=2):
        super().__init__()
        self.pad_idx = pad_idx
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.emb_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.out_dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, input_ids):
        emb = self.embedding(input_ids)
        emb = self.emb_dropout(emb)

        lengths = (input_ids != self.pad_idx).sum(dim=1).clamp(min=1)
        packed = nn.utils.rnn.pack_padded_sequence(
            emb, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, hidden = self.gru(packed)

        # hidden shape: [num_layers * 2, batch, hidden_dim]
        # For the last layer, take forward and backward hidden states.
        forward_hidden = hidden[-2]
        backward_hidden = hidden[-1]
        hidden = torch.cat((forward_hidden, backward_hidden), dim=1)

        hidden = self.out_dropout(hidden)
        return self.fc(hidden)


def word_dropout(input_ids, p=0.1, unk_id=UNK_ID, pad_id=PAD_ID):
    if p <= 0:
        return input_ids
    rand = torch.rand_like(input_ids, dtype=torch.float)
    mask = (rand < p) & (input_ids != pad_id)
    return torch.where(mask, torch.full_like(input_ids, unk_id), input_ids)


def main(lr=1e-3, embed_dim=64, hidden_dim=256, epochs=50, dropout=0.3,
         weight_decay=1e-2, patience=2, word_drop_p=0.1, label_smoothing=0.1,
         early_stop_patience=5):
    data = loadAndPrep()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Using device: {device}")
    print(f"Vocab size: {data['vocabSize']}")

    model = TextBiGRU(
        data["vocabSize"],
        embed_dim,
        hidden_dim,
        num_classes=6,
        dropout=dropout,
        num_layers=2,
    ).to(device)

    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    num_classes = 6
    counts = torch.bincount(data["trainY"], minlength=num_classes).float()
    class_weights = counts.sum() / (num_classes * counts.clamp(min=1))
    class_weights = class_weights.to(device)
    print(f"Class weights: {class_weights.tolist()}")
    loss_fn = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        opt,
        mode="min",
        factor=0.5,
        patience=patience,
    )

    best_f1 = 0.0
    best_state = None
    epochs_no_improve = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_total = 0

        for batch, targets in data["trainLoader"]:
            batch = batch.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            batch = word_dropout(batch, p=word_drop_p)

            opt.zero_grad()
            logits = model(batch)
            loss = loss_fn(logits, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            train_loss += loss.item() * targets.size(0)
            train_total += targets.size(0)

        model.eval()
        correct = 0
        total = 0
        total_loss = 0.0
        all_preds = []
        all_targets = []

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
                all_preds.append(preds.cpu())
                all_targets.append(targets.cpu())

        val_loss = total_loss / total
        val_acc = 100 * correct / total
        all_preds = torch.cat(all_preds).numpy()
        all_targets = torch.cat(all_targets).numpy()
        macro_f1 = f1_score(all_targets, all_preds, average="macro")

        scheduler.step(val_loss)

        if macro_f1 > best_f1:
            best_f1 = macro_f1
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        current_lr = opt.param_groups[0]["lr"]
        print(
            f"Epoch {epoch+1:3d} | "
            f"train loss {train_loss/train_total:.4f} | "
            f"val loss {val_loss:.4f} | "
            f"val acc {val_acc:.2f} | "
            f"macro F1 {macro_f1:.4f} | "
            f"best F1 {best_f1:.4f} | "
            f"lr {current_lr:.2e} | "
            f"no_improve {epochs_no_improve}"
        )

        if epochs_no_improve >= early_stop_patience:
            print(f"Early stop: no F1 improvement for {early_stop_patience} epochs.")
            break

    if best_state is not None:
        model.load_state_dict(best_state)


    print(f"Best val macro F1: {best_f1:.4f}")

    config = {
        "embed_dim": embed_dim,
        "hidden_dim": hidden_dim,
        "num_classes": 6,
        "dropout": dropout,
        "num_layers": 2,
    }
    torch.save({
        "type": "bigru",
        "state_dict": model.to("cpu").state_dict(),
        "mapping": data["mapping"],
        "vocabSize": data["vocabSize"],
        "config": config,
        }, "model2.pt")
    print("Model saved to model2.pt")


if __name__ == "__main__":
    main()
