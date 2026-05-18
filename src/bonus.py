import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns
from tqdm import tqdm

EMOTION_LABELS = ["sadness", "joy", "love", "anger", "fear", "surprise"]


def pickDevice():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class BertClassifier(nn.Module):
    def __init__(self, model_name="distilbert-base-uncased", num_classes=6, dropout=0.1):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        hidden_size = self.bert.config.hidden_size
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0, :]
        x = self.dropout(pooled)
        return self.classifier(x)


def prepare_bert_data():
    """Load and prepare data for BERT using HuggingFace tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    dataset = load_dataset("dair-ai/emotion", "split")

    tokenized = {}
    for split in ["train", "validation", "test"]:
        texts = [row["text"] for row in dataset[split]]
        labels = [row["label"] for row in dataset[split]]

        encoded = tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=64,
            return_tensors="pt"
        )

        tokenized[split] = {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "labels": torch.tensor(labels, dtype=torch.long)
        }

    return tokenized, tokenizer


def create_dataloaders(tokenized_data, batch_size=32):
    """Create DataLoaders from tokenized data."""
    loaders = {}
    for split in ["train", "validation", "test"]:
        dataset = torch.utils.data.TensorDataset(
            tokenized_data[split]["input_ids"],
            tokenized_data[split]["attention_mask"],
            tokenized_data[split]["labels"]
        )
        shuffle = (split == "train")
        loaders[split] = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle
        )
    return loaders


def train_bert(
    learning_rate=0.0001,
    batch_size=32,
    num_epochs=1,
    warmup_steps=500,
    dropout=0.1,
):
    """Fine-tune BERT on emotion dataset."""
    device = pickDevice()
    print(f"Using device: {device}")

    print("Loading and preparing data...")
    tokenized_data, tokenizer = prepare_bert_data()
    loaders = create_dataloaders(tokenized_data, batch_size=batch_size)

    print("Loading BERT model...")
    model = BertClassifier(dropout=dropout).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()

    num_training_steps = len(loaders["train"]) * num_epochs
    scheduler = optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1.0, total_iters=warmup_steps
    )

    best_val_acc = 0.0
    best_state = None
    train_losses, val_losses, val_accs = [], [], []

    print(f"\nTraining with lr={learning_rate}, batch_size={batch_size}, epochs={num_epochs}")
    print("=" * 60)

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0

        for input_ids, attention_mask, labels in tqdm(loaders["train"], desc=f"Epoch {epoch+1}/{num_epochs}"):
            input_ids = input_ids.to(device, non_blocking=True)
            attention_mask = attention_mask.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item()

        avg_train_loss = epoch_loss / len(loaders["train"])
        train_losses.append(avg_train_loss)

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for input_ids, attention_mask, labels in tqdm(loaders["validation"], desc="Validating", leave=False):
                input_ids = input_ids.to(device, non_blocking=True)
                attention_mask = attention_mask.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                logits = model(input_ids, attention_mask)
                loss = criterion(logits, labels)
                val_loss += loss.item()

                preds = logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        avg_val_loss = val_loss / len(loaders["validation"])
        val_acc = 100 * correct / total
        val_losses.append(avg_val_loss)
        val_accs.append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        print(f"Epoch {epoch+1:2d} | Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.2f}% (best {best_val_acc:.2f}%)")

    if best_state is not None:
        model.load_state_dict(best_state)

    print(f"\nBest validation accuracy: {best_val_acc:.2f}%")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")

    return model, loaders, device, train_losses, val_losses, val_accs


def evaluate_bert(model, loaders, device):
    """Evaluate BERT on test set and compute metrics."""
    model.eval()
    all_preds = []
    all_labels = []
    test_loss = 0.0

    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for input_ids, attention_mask, labels in loaders["test"]:
            input_ids = input_ids.to(device, non_blocking=True)
            attention_mask = attention_mask.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            test_loss += loss.item()

            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    test_acc = 100 * accuracy_score(all_labels, all_preds)
    avg_test_loss = test_loss / len(loaders["test"])

    print(f"\nTest Loss: {avg_test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.2f}%")
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=EMOTION_LABELS))

    return all_preds, all_labels, test_acc


def analyze_results(all_preds, all_labels, train_losses, val_losses, val_accs):
    """Perform analysis of BERT results with visualizations."""

    cm = confusion_matrix(all_labels, all_preds)

    print("\n" + "=" * 60)
    print("BERT Fine-tuning Analysis")
    print("=" * 60)

    print("\nConfusion Matrix:")
    print(cm)

    per_class_acc = cm.diagonal() / cm.sum(axis=1)
    print("\nPer-class Accuracy:")
    for label, acc in zip(EMOTION_LABELS, per_class_acc):
        print(f"  {label:12s}: {100*acc:.2f}%")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    ax.plot(range(1, len(train_losses) + 1), train_losses, label="Train Loss")
    ax.plot(range(1, len(val_losses) + 1), val_losses, label="Val Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training and Validation Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(range(1, len(val_accs) + 1), val_accs, marker='o', label="Val Accuracy")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Validation Accuracy Over Time")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=EMOTION_LABELS, yticklabels=EMOTION_LABELS, ax=ax)
    ax.set_title("Confusion Matrix")
    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")

    ax = axes[1, 1]
    ax.bar(EMOTION_LABELS, per_class_acc * 100)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Per-class Accuracy")
    ax.set_ylim([0, 105])
    for i, v in enumerate(per_class_acc * 100):
        ax.text(i, v + 1, f"{v:.1f}%", ha='center', va='bottom')
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig("bert_analysis.png", dpi=150, bbox_inches='tight')
    print("\nSaved analysis plots to bert_analysis.png")
    plt.show()


def main():
    """Main entry point for BERT fine-tuning."""
    print("\n" + "=" * 60)
    print("BERT Fine-tuning for Emotion Classification")
    print("=" * 60)

    device = pickDevice()
    
    print("Training BERT...")
    model, loaders, device, train_losses, val_losses, val_accs = train_bert(
        learning_rate=2e-5,
        batch_size=32,
        num_epochs=1,
        dropout=0.1,
    )

    print("\nEvaluating on test set...")
    all_preds, all_labels, test_acc = evaluate_bert(model, loaders, device)

    print("\nAnalyzing results...")
    analyze_results(all_preds, all_labels, train_losses, val_losses, val_accs)

    torch.save(model.state_dict(), "bert_model.pt")
    print("\nSaved BERT model to bert_model.pt")


if __name__ == "__main__":
    main()
