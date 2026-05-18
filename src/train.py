import torch
import torch.nn as nn

from data_prep import loadAndPrep, pickDevice
from model1 import TransformerClassifier


def main(lr = 0.001, n_heads = 4, n_layers = 4):
    data = loadAndPrep()
    device = pickDevice()
    print(f"Using device: {device}")
    print(f"Vocab size: {data['vocabSize']}")

    config = {"d_model": 128, "d_key": 32, "n_heads": n_heads, "mlp_factor": 4, "n_layers": n_layers, "n_classes": 6}
    model = TransformerClassifier(data["vocabSize"], **config).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    best_state = None
    per_epoch = 10

    for epoch in range(50):
        model.train()
        print(epoch)

        for batch, targets in data["trainLoader"]:
            batch = batch.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad()
            output = model(batch)
            loss = criterion(output, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        
        if (epoch + 1) % per_epoch == 0:
            model.eval()
            correct = 0
            total = 0
            total_loss = 0.0
            with torch.no_grad():
                for batch, targets in data["validationLoader"]:
                    batch = batch.to(device, non_blocking=True)
                    targets = targets.to(device, non_blocking=True)
                    outputs = model(batch)
                    loss = criterion(outputs, targets)
                    total_loss += loss.item() * targets.size(0)

                    _, predicted = torch.max(outputs, 1)
                    total += targets.size(0)
                    correct += (predicted == targets).sum().item()

                if best_acc < 100 * correct / total:
                    best_acc = 100 * correct / total
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

            print(f"Epoch {epoch+1}, ValLoss: {total_loss / total:.4f}, ValAcc: {100 * correct / total:.4f} (best {best_acc:.4f})")

    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"Best validation accuracy: {best_acc:.2f}%")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")

    torch.save({
        "state_dict": model.to("cpu").state_dict(),
        "mapping": data["mapping"],
        "vocabSize": data["vocabSize"],
        "config": config,
    }, "model.pt")
    print("Saved model.pt")


if __name__ == "__main__":
    main()
