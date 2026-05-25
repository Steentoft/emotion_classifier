import os
import urllib.request
import zipfile
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score

from data_prep import loadAndPrep, PAD_ID, UNK_ID, enc


PREFIX_PHRASES = [
    "i cant help but feel",
    "i have been feeling",
    "ive been feeling",
    "i have a feeling",
    "i get the feeling",
    "i am feeling",
    "im feeling",
    "i was feeling",
    "i don t feel",
    "i just feel",
    "i still feel",
    "i can feel",
    "i feel like",
    "i feel",
    "i felt",
    "i feeling",
]


def build_prefix_seqs(mapping):
    seqs = []
    for phrase in PREFIX_PHRASES:
        ids = enc.encode(phrase)
        remapped = [mapping.get(t, UNK_ID) for t in ids]
        seqs.append(remapped)
    seqs.sort(key=len, reverse=True)
    return seqs


def augment_batch(batch, prefix_seqs, p_word_drop=0.1, p_strip=0.4,
                  min_keep_tokens=3):
    batch = batch.clone()
    bsz, slen = batch.shape

    if p_strip > 0 and prefix_seqs:
        strip_mask = torch.rand(bsz) < p_strip
        for i in torch.where(strip_mask)[0].tolist():
            row = batch[i].tolist()
            row_len = sum(1 for t in row if t != PAD_ID)
            for seq in prefix_seqs:
                L = len(seq)
                if L < slen and row_len - L >= min_keep_tokens and row[:L] == seq:
                    new_row = row[L:] + [PAD_ID] * L
                    batch[i] = torch.tensor(new_row, dtype=batch.dtype)
                    break

    if p_word_drop > 0:
        non_pad = batch != PAD_ID
        lengths = non_pad.sum(dim=1)
        max_drops = (lengths - min_keep_tokens).clamp(min=0)
        drop_mask = (torch.rand(batch.shape) < p_word_drop) & non_pad
        # Cap drops per row to respect min_keep_tokens
        for i in range(bsz):
            row_drops = drop_mask[i].sum().item()
            cap = int(max_drops[i].item())
            if row_drops > cap:
                drop_idx = torch.where(drop_mask[i])[0]
                keep = drop_idx[torch.randperm(len(drop_idx))[:cap]]
                drop_mask[i] = False
                drop_mask[i, keep] = True
        batch[drop_mask] = UNK_ID

    return batch


GLOVE_URL = "https://nlp.stanford.edu/data/glove.6B.zip"
GLOVE_CACHE_DIR = os.path.expanduser("~/.cache/glove")
GLOVE_DIM_TO_FILENAME = {
    50: "glove.6B.50d.txt",
    100: "glove.6B.100d.txt",
    200: "glove.6B.200d.txt",
    300: "glove.6B.300d.txt",
}


def ensure_glove(dim=100):
    """Return local path to glove.6B.{dim}d.txt, downloading the zip if missing."""
    if dim not in GLOVE_DIM_TO_FILENAME:
        raise ValueError(f"GloVe dim must be one of {list(GLOVE_DIM_TO_FILENAME)}")
    os.makedirs(GLOVE_CACHE_DIR, exist_ok=True)
    target = os.path.join(GLOVE_CACHE_DIR, GLOVE_DIM_TO_FILENAME[dim])
    if os.path.exists(target):
        return target

    zip_path = os.path.join(GLOVE_CACHE_DIR, "glove.6B.zip")
    if not os.path.exists(zip_path):
        print(f"Downloading GloVe (~822MB) to {zip_path} ...")
        urllib.request.urlretrieve(GLOVE_URL, zip_path)
    print(f"Extracting {GLOVE_DIM_TO_FILENAME[dim]} ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extract(GLOVE_DIM_TO_FILENAME[dim], GLOVE_CACHE_DIR)
    return target


def load_glove(path, dim):
    vectors = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip().split(" ")
            if len(parts) != dim + 1:
                continue
            vectors[parts[0]] = torch.tensor([float(x) for x in parts[1:]],
                                             dtype=torch.float)
    return vectors


def build_pretrained_matrix(mapping, glove, dim):
    """Build an embedding matrix aligned with the remapped vocab.

    mapping: {tiktoken_token_id -> remapped_id (>=2)}.  PAD=0, UNK=1.
    Tokens are decoded via tiktoken, lowercased and stripped, then looked up.
    """
    vocab_size = len(mapping) + 2
    matrix = torch.empty(vocab_size, dim)
    nn.init.normal_(matrix, mean=0.0, std=0.1)
    matrix[PAD_ID] = 0.0

    hits = 0
    for tok_id, remap_id in mapping.items():
        piece = enc.decode([tok_id]).strip().lower()
        if piece and piece in glove:
            matrix[remap_id] = glove[piece]
            hits += 1
    print(f"GloVe coverage: {hits}/{len(mapping)} tokens "
          f"({100 * hits / max(1, len(mapping)):.1f}%)")
    return matrix


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
                 pad_idx=PAD_ID, dropout=0.3, num_layers=2,
                 pretrained_embeddings=None, freeze_embeddings=False):
        super().__init__()
        self.pad_idx = pad_idx
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        if pretrained_embeddings is not None:
            if pretrained_embeddings.shape != (vocab_size, embed_dim):
                raise ValueError(
                    f"pretrained_embeddings shape {tuple(pretrained_embeddings.shape)} "
                    f"does not match (vocab_size={vocab_size}, embed_dim={embed_dim})"
                )
            self.embedding.weight.data.copy_(pretrained_embeddings)
            if freeze_embeddings:
                self.embedding.weight.requires_grad = False
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




def main(lr=1e-3, embed_dim=64, hidden_dim=256, epochs=50, dropout=0.3,
         weight_decay=1e-2, patience=2,
         early_stop_patience=5, pretrained=False, glove_dim=100,
         freeze_embeddings=False, p_word_drop=0.1, p_prefix_strip=0.4):
    data = loadAndPrep()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Using device: {device}")
    print(f"Vocab size: {data['vocabSize']}")

    pretrained_matrix = None
    if pretrained:
        if embed_dim != glove_dim:
            print(f"[pretrained] overriding embed_dim {embed_dim} -> {glove_dim} "
                  f"to match GloVe")
            embed_dim = glove_dim
        glove_path = ensure_glove(glove_dim)
        print(f"Loading GloVe from {glove_path} ...")
        glove = load_glove(glove_path, glove_dim)
        pretrained_matrix = build_pretrained_matrix(data["mapping"], glove, glove_dim)
        print(f"freeze_embeddings={freeze_embeddings}")

    model = TextBiGRU(
        data["vocabSize"],
        embed_dim,
        hidden_dim,
        num_classes=6,
        dropout=dropout,
        num_layers=2,
        pretrained_embeddings=pretrained_matrix,
        freeze_embeddings=freeze_embeddings,
    ).to(device)

    prefix_seqs = build_prefix_seqs(data["mapping"])
    print(f"Built {len(prefix_seqs)} prefix-strip patterns "
          f"(p_strip={p_prefix_strip}, p_word_drop={p_word_drop})")

    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    num_classes = 6
    counts = torch.bincount(data["trainY"], minlength=num_classes).float()
    class_weights = counts.sum() / (num_classes * counts.clamp(min=1))
    class_weights = class_weights.to(device)
    print(f"Class weights: {class_weights.tolist()}")
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        opt,
        mode="min",
        factor=0.5,
        patience=patience,
    )

    best_f1 = 0.0
    best_state = None
    epochs_no_improve = 0

    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "val_acc": [],
        "macro_f1": [],
        "lr": [],
    }

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_total = 0

        for batch, targets in data["trainLoader"]:
            batch = augment_batch(batch, prefix_seqs,
                                  p_word_drop=p_word_drop,
                                  p_strip=p_prefix_strip)
            batch = batch.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

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
        avg_train_loss = train_loss / train_total
        history["epoch"].append(epoch + 1)
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["macro_f1"].append(macro_f1)
        history["lr"].append(current_lr)

        print(
            f"Epoch {epoch+1:3d} | "
            f"train loss {avg_train_loss:.4f} | "
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
    if pretrained:
        fname = "model2_glove_frozen.pt" if freeze_embeddings else "model2_glove_finetune.pt"
    else:
        fname = "model2.pt"

    os.makedirs("models", exist_ok=True)
    save_path = os.path.join("models", fname)
    if os.path.exists(save_path):
        os.remove(save_path)

    torch.save({
        "type": "bigru",
        "state_dict": model.to("cpu").state_dict(),
        "mapping": data["mapping"],
        "vocabSize": data["vocabSize"],
        "config": config,
        }, save_path)
    print(f"Model saved to {save_path}")

    plt.figure()
    plt.plot(history["epoch"], history["train_loss"], label="Training loss")
    plt.plot(history["epoch"], history["val_loss"], label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Model 2 training and validation loss")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
