// ── Architecture diagram helpers ─────────────────────────────────
#let arch-box(body, fill: rgb("#eef3fb"), stroke: rgb("#3b6ea8")) = block(
  width: 100%,
  inset: 6pt,
  radius: 4pt,
  fill: fill,
  stroke: 0.6pt + stroke,
  align(center, body),
)

#let arch-arrow = align(center, text(size: 14pt)[#sym.arrow.b])

#let arch-diagram(title, layers) = figure(
  caption: title,
  block(
    width: 70%,
    stack(
      dir: ttb,
      spacing: 4pt,
      ..layers
        .enumerate()
        .map(((i, l)) => if i == 0 { l } else { stack(dir: ttb, spacing: 4pt, arch-arrow, l) })
    )
  )
)

// ── Model 1 diagram ──────────────────────────────────────────────
== Model 1 architecture
#arch-diagram("Model 1: Transformer encoder", (
  arch-box(raw("Input token ids [B, L]"), fill: rgb("#f2f2f2"), stroke: gray),
  arch-box([Token Embedding \ (vocab, d_model = 128)]),
  arch-box([
    *Transformer Block × 4* \
    LayerNorm #sym.arrow.r MultiHeadSelfAttention (4 heads) #sym.arrow.r residual \
    LayerNorm #sym.arrow.r MLP (Linear #sym.arrow.r SiLU #sym.arrow.r Linear, ×4 expand) #sym.arrow.r residual
  ]),
  arch-box([Final LayerNorm]),
  arch-box(raw("Mean Pool over sequence dim -> [B, 128]")),
  arch-box([Classifier MLP \ Linear(128 #sym.arrow.r 128) #sym.arrow.r SiLU #sym.arrow.r Linear(128 #sym.arrow.r 6)]),
  arch-box(raw("Logits [B, 6]"), fill: rgb("#e9f7ec"), stroke: rgb("#3a8a4f")),
))

// ── Model 2 diagram ──────────────────────────────────────────────
== Model 2 architecture
#arch-diagram("Model 2: Bidirectional GRU", (
  arch-box(raw("Input token ids [B, L]"), fill: rgb("#f2f2f2"), stroke: gray),
  arch-box([Token Embedding \ (vocab, embed_dim = 64, padding_idx = 0)]),
  arch-box([Dropout (p = 0.3)]),
  arch-box([`pack_padded_sequence` \ (skip PAD tokens)]),
  arch-box([
    *Bidirectional GRU* \
    num_layers = 2, hidden_dim = 256 \
    inter-layer dropout = 0.3
  ]),
  arch-box([
    Take last-layer hidden: \
    forward `[B, 256]` #h(0.4em) #sym.plus.circle #h(0.4em) backward `[B, 256]` \
    #sym.arrow.r concat `[B, 512]`
  ]),
  arch-box([Dropout (p = 0.3)]),
  arch-box([Linear(512 #sym.arrow.r 6)]),
  arch-box(raw("Logits [B, 6]"), fill: rgb("#e9f7ec"), stroke: rgb("#3a8a4f")),
))
