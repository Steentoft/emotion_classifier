import os

import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from data_prep import loadAndPrep, pickDevice
from model1 import TransformerClassifier


def main(lr=0.001, n_heads=4, n_layers=4, epochs=50):
    data = loadAndPrep()
    device = pickDevice()
    print(f"Using device: {device}")
    print(f"Vocab size: {data['vocabSize']}")

    config = {
        "d_model": 128,
        "d_key": 32,
        "n_heads": n_heads,
        "mlp_factor": 4,
        "n_layers": n_layers,
        "n_classes": 6,
    }
    model = TransformerClassifier(data["vocabSize"], **config).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    best_state = None

    train_losses = []
    val_losses = []
    epochs_logged = []

    for epoch in range(epochs):
        model.train()
        train_loss_sum = 0.0
        train_total = 0

        for batch, targets in data["trainLoader"]:
            batch = batch.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad()
            output = model(batch)
            loss = criterion(output, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss_sum += loss.item() * targets.size(0)
            train_total += targets.size(0)

        train_loss = train_loss_sum / train_total

        model.eval()
        correct = 0
        total = 0
        val_loss_sum = 0.0

        with torch.no_grad():
            for batch, targets in data["validationLoader"]:
                batch = batch.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)

                outputs = model(batch)
                loss = criterion(outputs, targets)
                val_loss_sum += loss.item() * targets.size(0)

                _, predicted = torch.max(outputs, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

        val_loss = val_loss_sum / total
        val_acc = 100 * correct / total

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        epochs_logged.append(epoch + 1)

        if best_acc < val_acc:
            best_acc = val_acc
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }

        print(
            f"Epoch {epoch + 1:3d}/{epochs} | "
            f"train loss {train_loss:.4f} | "
            f"val loss {val_loss:.4f} | "
            f"val acc {val_acc:.2f}% | "
            f"best {best_acc:.2f}%"
        )

    if best_state is not None:
        model.load_state_dict(best_state)

    print(f"Best validation accuracy: {best_acc:.2f}%")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")

    os.makedirs("models", exist_ok=True)
    save_path = os.path.join("models", "model.pt")
    if os.path.exists(save_path):
        os.remove(save_path)
    torch.save({
        "type": "transformer",
        "state_dict": model.to("cpu").state_dict(),
        "mapping": data["mapping"],
        "vocabSize": data["vocabSize"],
        "config": config,
    }, save_path)
    print(f"Saved {save_path}")

    plt.figure()
    plt.plot(epochs_logged, train_losses, label="Training loss")
    plt.plot(epochs_logged, val_losses, label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Model 1 training and validation loss")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
