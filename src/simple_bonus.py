import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns
from tqdm import tqdm
from data_prep import pickDevice

EMOTION_LABELS = ["sadness", "joy", "love", "anger", "fear", "surprise"]


def main():
    device = pickDevice()
    print(f"\nUsing device: {device}\n")

    # Load and tokenize data
    print("Loading data...")
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    dataset = load_dataset("dair-ai/emotion", "split")

    loaders = {}
    for split in ["train", "validation", "test"]:
        texts = [row["text"] for row in dataset[split]]
        labels = torch.tensor([row["label"] for row in dataset[split]], dtype=torch.long)

        encoded = tokenizer(texts, padding="max_length", truncation=True, max_length=64, return_tensors="pt")
        ds = torch.utils.data.TensorDataset(encoded["input_ids"], encoded["attention_mask"], labels)
        shuffle = (split == "train")
        loaders[split] = torch.utils.data.DataLoader(ds, batch_size=32, shuffle=shuffle)

    # Model
    print("Loading DistilBERT...")
    bert = AutoModel.from_pretrained("distilbert-base-uncased")
    model = nn.Sequential(
        bert,
        nn.Linear(bert.config.hidden_size, 6)
    )
    model.to(device)

    # Training
    optimizer = optim.Adam(model.parameters(), lr=2e-5)
    criterion = nn.CrossEntropyLoss()
    epochs = 1
    best_val_acc = 0.0
    train_losses, val_losses, val_accs = [], [], []

    print(f"Training for {epochs} epoch(s)...\n")
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        for input_ids, attention_mask, labels in tqdm(loaders["train"], desc=f"Epoch {epoch+1}"):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model[0](input_ids=input_ids, attention_mask=attention_mask)
            pooled = outputs.last_hidden_state[:, 0, :]
            logits = model[1](pooled)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()

        avg_train_loss = epoch_loss / len(loaders["train"])
        train_losses.append(avg_train_loss)

        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for input_ids, attention_mask, labels in tqdm(loaders["validation"], desc="Validating", leave=False):
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                labels = labels.to(device)

                outputs = model[0](input_ids=input_ids, attention_mask=attention_mask)
                pooled = outputs.last_hidden_state[:, 0, :]
                logits = model[1](pooled)
                loss = criterion(logits, labels)
                val_loss += loss.item()

                correct += (logits.argmax(dim=1) == labels).sum().item()
                total += labels.size(0)

        avg_val_loss = val_loss / len(loaders["validation"])
        val_acc = 100 * correct / total
        val_losses.append(avg_val_loss)
        val_accs.append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc

        print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.2f}%\n")

    # Test evaluation
    print("Evaluating on test set...")
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for input_ids, attention_mask, labels in loaders["test"]:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            outputs = model[0](input_ids=input_ids, attention_mask=attention_mask)
            pooled = outputs.last_hidden_state[:, 0, :]
            logits = model[1](pooled)

            all_preds.extend(logits.argmax(dim=1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    test_acc = 100 * accuracy_score(all_labels, all_preds)
    print(f"\nTest Accuracy: {test_acc:.2f}%")
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=EMOTION_LABELS))

    # Analysis
    cm = confusion_matrix(all_labels, all_preds)
    per_class_acc = cm.diagonal() / cm.sum(axis=1)

    print("\nPer-class Accuracy:")
    for label, acc in zip(EMOTION_LABELS, per_class_acc):
        print(f"  {label:12s}: {100*acc:.2f}%")

    # Plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].plot(range(1, len(train_losses) + 1), train_losses, label="Train", marker='o')
    axes[0, 0].plot(range(1, len(val_losses) + 1), val_losses, label="Val", marker='o')
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].set_title("Loss")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(range(1, len(val_accs) + 1), val_accs, marker='o', color='green')
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Accuracy (%)")
    axes[0, 1].set_title("Validation Accuracy")
    axes[0, 1].grid(True, alpha=0.3)

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=EMOTION_LABELS, yticklabels=EMOTION_LABELS, ax=axes[1, 0])
    axes[1, 0].set_title("Confusion Matrix")
    axes[1, 0].set_ylabel("True")
    axes[1, 0].set_xlabel("Predicted")

    axes[1, 1].bar(EMOTION_LABELS, per_class_acc * 100)
    axes[1, 1].set_ylabel("Accuracy (%)")
    axes[1, 1].set_title("Per-class Accuracy")
    axes[1, 1].set_ylim([0, 105])
    plt.setp(axes[1, 1].xaxis.get_majorticklabels(), rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig("simple_bert_analysis.png", dpi=150, bbox_inches='tight')
    print("\nSaved plots to simple_bert_analysis.png")
    plt.show()

    torch.save(model.state_dict(), "simple_bert_model.pt")
    print("Saved model to simple_bert_model.pt")


if __name__ == "__main__":
    main()
