import argparse

import torch

from data_prep import (
    encodeTextTiktoken,
    applyRemap,
    convertLength,
    convert2Tensor,
    enc,
    LABELS,
    MAX_LEN,
    PAD_ID,
    UNK_ID,
)
from predict import loadModel


def tokensToText(ids, mapping):
    """Reverse mapping then decode via tiktoken. PAD/UNK shown as markers."""
    inv = {v: k for k, v in mapping.items()}
    pieces = []
    for tid in ids:
        if tid == PAD_ID:
            continue
        if tid == UNK_ID:
            pieces.append("<UNK>")
            continue
        if tid in inv:
            pieces.append(enc.decode([inv[tid]]))
        else:
            pieces.append("<?>")
    return "".join(pieces)


def encode(text, mapping, maxLength=MAX_LEN):
    raw = encodeTextTiktoken(text)
    remapped = applyRemap([raw], mapping)
    padded = convertLength(remapped, maxLength)
    return raw, remapped[0], convert2Tensor(padded)


@torch.no_grad()
def probsFor(model, x, device):
    logits = model(x.to(device))
    return torch.softmax(logits, dim=-1)[0].cpu()


def predictFull(model, mapping, text, device, maxLength=MAX_LEN):
    raw, remapped, x = encode(text, mapping, maxLength)
    p = probsFor(model, x, device)
    idx = int(torch.argmax(p).item())
    unk = sum(1 for t in remapped if t == UNK_ID)
    return {
        "label": LABELS[idx],
        "conf": float(p[idx]),
        "probs": p,
        "raw": raw,
        "remapped": remapped,
        "unk": unk,
    }


def printProbs(p):
    pairs = sorted(zip(LABELS, p.tolist()), key=lambda kv: -kv[1])
    print("  " + "  ".join(f"{l}={v:.2f}" for l, v in pairs))


# ----- Occlusion attribution -----
@torch.no_grad()
def occlusion(model, mapping, text, device, maxLength=MAX_LEN):
    """Drop each token (replace with PAD), measure drop in predicted class prob.
    Big drop = token mattered for predicted class."""
    raw, remapped, x = encode(text, mapping, maxLength)
    base = probsFor(model, x, device)
    cls = int(torch.argmax(base).item())
    baseP = float(base[cls])

    pieces = [enc.decode([t]) for t in raw]
    scores = []
    for i, _ in enumerate(raw):
        masked = list(remapped)
        masked[i] = PAD_ID
        padded = convertLength([masked], maxLength)
        xi = convert2Tensor(padded)
        pi = probsFor(model, xi, device)
        scores.append(baseP - float(pi[cls]))
    return LABELS[cls], baseP, list(zip(pieces, scores))


def printAttribution(label, baseP, scored, topK=8):
    print(f"  pred={label}  p={baseP:.3f}  (token -> drop in p when removed)")
    rank = sorted(enumerate(scored), key=lambda kv: -kv[1][1])
    for i, (tok, s) in rank[:topK]:
        marker = "++" if s > 0 else ("--" if s < 0 else "  ")
        print(f"    {marker} {s:+.3f}  '{tok}'")


# ----- Word swap helper -----
def swap(text, old, new):
    """Word-level substitution preserving spaces. Case-insensitive."""
    import re
    pattern = re.compile(rf"\b{re.escape(old)}\b", re.IGNORECASE)
    return pattern.sub(new, text)


def compare(model, mapping, before, after, device):
    a = predictFull(model, mapping, before, device)
    b = predictFull(model, mapping, after, device)
    print(f"BEFORE: {before}")
    print(f"  -> {a['label']} ({a['conf']:.2%})")
    printProbs(a["probs"])
    print(f"AFTER:  {after}")
    print(f"  -> {b['label']} ({b['conf']:.2%})")
    printProbs(b["probs"])
    flipped = a["label"] != b["label"]
    print(f"  flipped={flipped}")
    return a, b


# ----- Failure case loader -----
def loadFailures(path):
    """Parse eval_results/*_failures.txt produced by eval_all.py."""
    cases = []
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.startswith("[true="):
                continue
            head, _, text = line.partition("] ")
            head = head[1:]  # strip leading [
            parts = head.split()
            true = parts[0].split("=")[1]
            pred = parts[1].split("=")[1]
            cases.append({"true": true, "pred": pred, "text": text})
    return cases


def inspectFailures(model, mapping, device, path, n=5, withAttribution=True):
    cases = loadFailures(path)
    print(f"Loaded {len(cases)} failures from {path}; showing first {n}")
    for c in cases[:n]:
        print("-" * 70)
        print(f"true={c['true']:8s} pred-in-file={c['pred']:8s}")
        print(f"text: {c['text']}")
        r = predictFull(model, mapping, c["text"], device)
        print(f"now-> {r['label']} ({r['conf']:.2%})  unk={r['unk']}")
        printProbs(r["probs"])
        if withAttribution:
            label, baseP, scored = occlusion(model, mapping, c["text"], device)
            printAttribution(label, baseP, scored)


# ----- REPL -----
HELP = """\
Commands:
  <text>                    classify text
  :a <text>                 classify + occlusion attribution
  :swap <old>|<new>|<text>  classify text, then with old->new replaced
  :fail <path> [n]          inspect first n failures from file
  :decode <ids>             decode space-separated token ids back to text
  :h | :help                this help
  :q | empty                quit
"""


def repl(model, mapping, device, failPathDefault=None):
    print(HELP)
    while True:
        try:
            line = input("> ").strip()
        except EOFError:
            break
        if not line or line in (":q", ":quit"):
            break
        if line in (":h", ":help"):
            print(HELP)
            continue
        if line.startswith(":a "):
            text = line[3:].strip()
            label, baseP, scored = occlusion(model, mapping, text, device)
            printAttribution(label, baseP, scored)
            continue
        if line.startswith(":swap "):
            body = line[6:]
            try:
                old, new, text = body.split("|", 2)
            except ValueError:
                print("usage: :swap old|new|text")
                continue
            compare(model, mapping, text.strip(), swap(text.strip(), old.strip(), new.strip()), device)
            continue
        if line.startswith(":fail"):
            parts = line.split()
            path = parts[1] if len(parts) > 1 else failPathDefault
            if path is None:
                print("need path: :fail <path> [n]")
                continue
            n = int(parts[2]) if len(parts) > 2 else 5
            inspectFailures(model, mapping, device, path, n=n)
            continue
        if line.startswith(":decode "):
            ids = [int(x) for x in line[8:].split()]
            print(tokensToText(ids, mapping))
            continue
        r = predictFull(model, mapping, line, device)
        print(f"-> {r['label']} ({r['conf']:.2%})  unk={r['unk']}")
        printProbs(r["probs"])


def main():
    ap = argparse.ArgumentParser(description="Task 4 analysis: introspect emotion classifier predictions.")
    ap.add_argument("--model", default="models/model2.pt", help="path to .pt checkpoint")
    ap.add_argument("--failures", default="eval_results/model2_failures.txt",
                    help="default failures file for :fail command")
    ap.add_argument("--inspect", type=int, default=0,
                    help="if >0, non-interactively dump that many failure cases and exit")
    ap.add_argument("--text", default=None, help="classify a single text (with attribution) and exit")
    args = ap.parse_args()

    model, mapping, device = loadModel(args.model)
    print(f"Loaded {args.model} on {device}; vocab(+PAD,UNK)={len(mapping) + 2}")

    if args.text is not None:
        r = predictFull(model, mapping, args.text, device)
        print(f"-> {r['label']} ({r['conf']:.2%})  unk={r['unk']}")
        printProbs(r["probs"])
        label, baseP, scored = occlusion(model, mapping, args.text, device)
        printAttribution(label, baseP, scored)
        return
    if args.inspect > 0:
        inspectFailures(model, mapping, device, args.failures, n=args.inspect)
        return
    repl(model, mapping, device, failPathDefault=args.failures)


if __name__ == "__main__":
    main()
