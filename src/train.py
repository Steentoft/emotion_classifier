import torch
import torch.nn as nn

from data_prep import loadAndPrep, pickDevice
from model1 import TransformerClassifier


def main():
    data = loadAndPrep()
    device = pickDevice()
    print(f"Using device: {device}")
    print(f"Vocab size: {data['vocabSize']}")

    trainX = data["trainX"].to(device)
    valX = data["valX"].to(device)
    trainY = data["trainY"].to(device)
    valY = data["valY"].to(device)

    config = {"d_model": 8, "d_key": 8, "n_heads": 4, "mlp_factor": 4, "n_layers": 4, "n_classes": 6}
    model = TransformerClassifier(data["vocabSize"], **config).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    best_state = None

    for epoch in range(500):
        model.train()
        optimizer.zero_grad()
        loss = criterion(model(trainX), trainY)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        model.eval()
        with torch.no_grad():
            vp = model(valX)
            vacc = (torch.argmax(vp, dim=1) == valY).float().mean().item()
        if vacc > best_acc:
            best_acc = vacc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}, ValAcc: {vacc:.4f} (best {best_acc:.4f})")

    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"Best validation accuracy: {100 * best_acc:.2f}%")
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
