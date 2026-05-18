import os

import torch
from datasets import load_dataset

from data_prep import (
    encodeTextTiktoken, applyRemap, convertLength, convert2Tensor,
    LABELS, MAX_LEN,
)
from predict import loadModel


MODEL_PATHS = [
    "models/model.pt",
    "models/model2.pt",
    "models/model2_glove_frozen.pt",
    "models/model2_glove_finetune.pt",
]


FAIL_DIR = "eval_results"


def eval_one(path, texts, ys):
    print(f"\n{'='*60}\nEvaluating: {path}\n{'='*60}")
    if not os.path.exists(path):
        print(f"[skip] file not found")
        return None

    model, mapping, device = loadModel(path)
    print(f"device: {device}")

    enc = [encodeTextTiktoken(t) for t in texts]
    remap = applyRemap(enc, mapping)
    pad = convertLength(remap, MAX_LEN)
    X = convert2Tensor(pad).to(device)

    bs = 256
    preds = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            logits = model(X[i:i + bs])
            preds.append(logits.argmax(-1).cpu())
    preds = torch.cat(preds)

    acc = (preds == ys).float().mean().item()
    print(f"Test accuracy: {acc:.4f}  ({int((preds==ys).sum())}/{len(ys)})")

    os.makedirs(FAIL_DIR, exist_ok=True)
    base = os.path.splitext(os.path.basename(path))[0]
    fail_path = os.path.join(FAIL_DIR, f"{base}_failures.txt")
    n_fail = 0
    with open(fail_path, "w", encoding="utf-8") as f:
        f.write(f"# Failures for {path}\n")
        f.write(f"# Test acc: {acc:.4f}\n")
        f.write("# Format: [true=LABEL pred=LABEL] text\n\n")
        for i, (p, y) in enumerate(zip(preds.tolist(), ys.tolist())):
            if p != y:
                f.write(f"[true={LABELS[y]:<8} pred={LABELS[p]:<8}] {texts[i]}\n")
                n_fail += 1
    print(f"Wrote {n_fail} failures to {fail_path}")

    n_classes = len(LABELS)
    print(f"\n{'class':<10} {'support':>7} {'prec':>6} {'recall':>7} {'f1':>6}")
    macro_f1 = 0.0
    for c in range(n_classes):
        tp = int(((preds == c) & (ys == c)).sum())
        fp = int(((preds == c) & (ys != c)).sum())
        fn = int(((preds != c) & (ys == c)).sum())
        support = int((ys == c).sum())
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        macro_f1 += f1
        print(f"{LABELS[c]:<10} {support:>7} {prec:>6.3f} {rec:>7.3f} {f1:>6.3f}")
    macro_f1 /= n_classes
    print(f"\nMacro F1: {macro_f1:.4f}")

    cm = torch.zeros(n_classes, n_classes, dtype=torch.long)
    for t, p in zip(ys.tolist(), preds.tolist()):
        cm[t, p] += 1
    print("\nConfusion matrix (rows=true, cols=pred):")
    print("          " + " ".join(f"{l[:5]:>6}" for l in LABELS))
    for i, l in enumerate(LABELS):
        row = " ".join(f"{int(cm[i, j]):>6}" for j in range(n_classes))
        print(f"{l[:8]:<10}{row}")

    return {"path": path, "acc": acc, "macro_f1": macro_f1}


def main():
    ds = load_dataset("dair-ai/emotion", "split")
    test = ds["test"]
    texts = [r["text"] for r in test]
    ys = torch.tensor([r["label"] for r in test], dtype=torch.long)

    results = []
    for path in MODEL_PATHS:
        r = eval_one(path, texts, ys)
        if r is not None:
            results.append(r)

    print(f"\n{'='*60}\nSUMMARY\n{'='*60}")
    print(f"{'model':<45} {'acc':>7} {'macroF1':>9}")
    for r in results:
        print(f"{r['path']:<45} {r['acc']:>7.4f} {r['macro_f1']:>9.4f}")


if __name__ == "__main__":
    main()
