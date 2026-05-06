import torch

from data_prep import (
    encodeTextTiktoken,
    applyRemap,
    convertLength,
    convert2Tensor,
    pickDevice,
    ownEncoder,
    LABELS,
    MAX_LEN,
)
from model_1 import TransformerClassifier


def loadModel(path="model.pt", device=None):
    if device is None:
        device = pickDevice()
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = TransformerClassifier(ckpt["vocabSize"], **ckpt["config"])
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()
    return model, ckpt["mapping"], device


def predict(model, mapping, text, device, maxLength=MAX_LEN, debug=True):
    raw = encodeTextTiktoken(text)
    if debug:
        pieces = [ownEncoder.decode([t]) for t in raw]
        flags = ["UNK" if t not in mapping else "ok" for t in raw]
        print(f"raw ids:  {raw}")
        print(f"pieces:   {pieces}")
        print(f"in vocab: {flags}")
    remapped = applyRemap([raw], mapping)
    padded = convertLength(remapped, maxLength)
    x = convert2Tensor(padded).to(device)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=-1)[0]
        idx = int(torch.argmax(probs).item())
    unk = sum(1 for t in remapped[0] if t == 1)
    return LABELS[idx], float(probs[idx].item()), unk


def main():
    model, mapping, device = loadModel()
    print(f"Using device: {device}")
    while True:
        text = input("Sentence (empty to quit): ").strip()
        if not text:
            break
        label, conf, unk = predict(model, mapping, text, device)
        print(f"{label} ({conf:.2%})  unknown_tokens={unk}")


if __name__ == "__main__":
    main()