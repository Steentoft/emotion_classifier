// Sections for Task 4 (Analysis) and Task 5 (BERT fine-tuning bonus).
// Paste into the main report (or `#include "analysis_section.typ"`)
// in place of the empty `= Analysis` and `= Fine-tuning a pre-trained
// language model (Bonus)` headings.

= Analysis

For this analysis we focus on Model 2 (the Bidirectional GRU), since it is our
better-performing from-scratch model. To make the model usable interactively we
added `src/analysis.py`, which loads any of our checkpoints and exposes a small
REPL on top of `predict.py`. The REPL takes a sentence from standard input and
prints the predicted label together with the full probability distribution. It
also implements three tools we use throughout this section:

- *Token-level occlusion attribution*: for a given input we re-run the model
  $L$ times, each time replacing one token with `<PAD>`, and report the drop in
  the predicted class probability. Large positive drops mean the token was
  load-bearing for the prediction; near-zero or negative drops mean the model
  did not depend on it (or even relied on it slightly for a different class).
- *Word-swap helper* (`:swap old|new|text`) for minimal-edit experiments where
  we try to flip a wrong prediction by changing a single word, or break a
  correct prediction the same way.
- *Failure inspector* (`:fail eval_results/model2_failures.txt`) that reads the
  failure file produced by `eval_all.py`, re-predicts each row, and prints the
  occlusion attribution next to it.

We also implemented the hint: `tokensToText` inverts the training mapping and
decodes the tiktoken pieces back to a string, which is useful for sanity
checking what the model actually sees after vocabulary remapping and `<UNK>`
substitution.

== Inspecting failure cases

Running `python src/analysis.py --inspect 8` on
`eval_results/model2_failures.txt` reproduces the test-set mistakes and prints
the per-token attribution for each. A clear pattern emerges: most failures are
driven by *one or two high-signal tokens* that override the rest of the
sentence. Three representative examples:

#table(
  columns: (auto, auto, auto, auto),
  align: (left, left, left, left),
  [*Text*], [*True*], [*Pred*], [*Top attribution*],
  [#text(size: 9pt)[i don t feel particularly agitated]],
  [fear], [anger], [`agitated` (+0.64)],
  [#text(size: 9pt)[i feel very honoured ... im curious do any of you read magazines ...]],
  [joy], [surprise], [`curious` (+0.55)],
  [#text(size: 9pt)[i feel a bit stressed even though all the things i have going on are fun]],
  [anger], [sadness], [`stressed` (+0.24)],
)

In every case the model latches onto a single emotion-loaded word and ignores
the rest of the sentence — including explicit negation (`don t`), softening
(`a bit ... are fun`) and topic-change discourse (`im curious do any of you`).
This is consistent with the bag-of-words flavour of our GRU pooling and with
how the augmentation/training regime rewards keyword sensitivity.

We see three broad categories of failures:

  1. *Negation ignored.* The token `agitated` maps almost deterministically to
     anger/fear, even when wrapped in `i don t feel ... agitated`.
  2. *Label ambiguity in the dataset.* Several "errors" are arguably correct.
     `i feel like i am in paradise kissing those sweet lips` is labelled
     *joy* in the dataset but predicted *love* with 97% confidence — the
     model is not wrong, the gold label is debatable. `i feel a bit stressed
     even though all the things i have going on are fun` is labelled *anger*
     but reads much more like *sadness/fear* to a human.
  3. *Spurious cues winning over weak global context.* In the long
     `i explain why i clung ... excitement i should have been feeling ...`
     sentence the top contributors are `and`, `immature`, `who`, `many` —
     i.e. function words and a single content word rather than the
     joy-signalling `excitement`. The model has no strong joy cue to anchor on
     and drifts to *surprise*.

== Minimal-edit experiments

Using the swap helper we tried to (a) fix a wrong prediction with as little
text change as possible and (b) break a correct prediction the same way.

*Fixing a failure.* On `i don t feel particularly agitated` (true fear, pred
anger 70%) the only change needed was replacing `agitated` with a less
anger-coded fear word:

```
BEFORE: i don t feel particularly agitated   -> anger (70%)
AFTER:  i don t feel particularly scared     -> fear  (99.97%)
```

*Breaking a correct prediction.* On `i feel so happy today` (joy 99.8%) we
changed `happy` to `weird`:

```
BEFORE: i feel so happy today   -> joy      (99.82%)
AFTER:  i feel weird today      -> surprise (83.72%)
```

A single-word edit is enough in both directions, which reinforces the
keyword-driven story above.

*Negation handling.* The most striking finding was that the model essentially
does not handle negation:

```
i am happy       -> joy (49.4%)
i am not happy   -> joy (60.7%)
```

Not only does the label not flip, the confidence in joy actually *increases*.
The model treats `happy` as a joy signal regardless of the surrounding `not`.
This is a real, reproducible failure mode and a fundamental limitation of a
mean-pooled bidirectional GRU trained on this dataset, where almost no training
sentences are constructed as overt negations of an emotion word.

== Does the model generalize to longer texts?

We took a short anchor sentence and padded it with semantically consistent
filler beyond the 30-token training window:

```
short: "i feel sad"                                  -> sadness (99.5%)
long : "i feel sad and lonely while sitting on the
        porch watching the rain fall over the empty
        street ..." (repeated 3x)                    -> sadness (99.9%)
```

The prediction is stable and confidence even rises slightly, which is what we
would expect — the truncation to `MAX_LEN = 30` means the model never sees the
long tail anyway, but as long as the leading 30 tokens contain the high-signal
emotion word the answer is preserved. The model "generalises" to longer text in
the trivial sense that nothing in the first 30 tokens has to change. We did not
see any case where appending neutral filler flipped a confident prediction.

== Summary of findings

- *Predictions are dominated by a small number of emotion-loaded tokens.* For
  almost every input we inspected, one or two words account for the bulk of the
  occlusion score; the rest of the sentence is essentially context the model
  ignores.
- *Indicative words per class are easy to read off.* `happy`, `excited`,
  `delighted` push joy; `sad`, `lonely`, `groggy` push sadness; `love`,
  `liked`, `sweet`, `kissing` push love; `scared`, `frantic`, `overwhelmed`
  push fear; `agitated`, `furious`, `stressed` push anger; `weird`, `curious`,
  `dazed`, `surprised` push surprise.
- *The model does not handle negation*, which we showed directly with
  `i am not happy` still being classified as joy with higher confidence than
  the un-negated version.
- *Length is not an issue in itself*, but only because the input is truncated
  to 30 tokens before the model sees it. Truncation can itself cause failures
  on long tweets where the emotional cue sits past token 30.
- *A non-trivial fraction of "failures" are label disagreements*, not model
  errors. This caps the achievable test accuracy and makes per-class macro F1
  the more honest metric.

The model's predictions match our intuition on short, keyword-rich sentences
and diverge from it precisely where the dataset itself is weakest: long
narrative tweets, negation, and sentences where the emotion is implied rather
than named.

= Fine-tuning a pre-trained language model (Bonus)

For the bonus we fine-tuned `distilbert-base-uncased` from the HuggingFace
hub on the same Emotion dataset. DistilBERT is a 6-layer distilled version of
BERT-base (66M parameters, ~40% smaller and ~60% faster than BERT-base) which
keeps most of BERT's quality and trains comfortably on a single laptop GPU.
The code lives in `src/bonus.py`.

== Setup

- *Tokenizer*: the model-specific `AutoTokenizer.from_pretrained("distilbert-base-uncased")`.
  This is the important contrast with our from-scratch models — DistilBERT
  brings its own WordPiece vocabulary, so we no longer use tiktoken or our
  hand-built mapping. We pad/truncate to `max_length = 64`, comfortably larger
  than the 30-token window we use for Models 1 and 2.
- *Architecture*: the pretrained DistilBERT encoder followed by dropout and a
  single linear classifier on top of the `[CLS]` pooled hidden state
  (`outputs.last_hidden_state[:, 0, :]`).
- *Loss / optimizer*: standard `CrossEntropyLoss` (unweighted — the pretrained
  features turn out strong enough that class-weighting is not needed) with
  `AdamW`, weight decay $1 times 10^(-2)$, gradient clipping at norm 1.0, and a
  short linear warmup schedule.

```python
class BertClassifier(nn.Module):
    def __init__(self, model_name="distilbert-base-uncased",
                 num_classes=6, dropout=0.1):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(
            self.bert.config.hidden_size, num_classes
        )

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids,
                        attention_mask=attention_mask)
        pooled = out.last_hidden_state[:, 0, :]
        return self.classifier(self.dropout(pooled))
```

== Hyperparameter tuning

Fine-tuning a pretrained encoder is far more sensitive to learning rate than
training Model 2 from scratch — too large and the pretrained weights collapse,
too small and the head never learns. We swept a small grid on the validation
set:

- learning rate: ${5 times 10^(-5), 2 times 10^(-5), 1 times 10^(-5)}$,
- batch size: ${16, 32, 64}$,
- dropout: ${0.1, 0.2, 0.3}$,
- epochs: ${1, 2, 3}$.

The final configuration we adopted was learning rate $2 times 10^(-5)$, batch
size 32, dropout 0.1, weight decay $1 times 10^(-2)$, 1 epoch. With this setup
the validation loss already started rising at epoch 2, so additional epochs
hurt — consistent with the usual BERT fine-tuning recipe of 1–3 epochs.

== Results

#figure(
  caption: [DistilBERT fine-tuning analysis: loss curves, validation
            accuracy, test-set confusion matrix and per-class accuracy.
            Produced by `src/bonus.py`.],
  image("bert_analysis.png", width: 100%),
)

After fine-tuning, DistilBERT reaches *92.05% test accuracy* (1841/2000
correct), compared with 89.25% for our from-scratch Bidirectional GRU and
90.15% for the GRU initialized with GloVe. The gain over the GRU is real but
smaller than one might naively expect for "BERT vs. from scratch", which
reflects two things: (a) our GRU is already a relatively strong baseline once
we add class-weighted loss, augmentation and early stopping, and (b) the
Emotion dataset is small (16k training rows) and the labels are noisy in
exactly the corners where a stronger model would otherwise pull ahead.

The per-class picture is more interesting than the headline number:

#table(
  columns: 4,
  align: (left, center, center, center),
  [*Class*], [*DistilBERT*], [*Model 2 (scratch)*], [*Support*],
  [sadness ], [94.1%], [—], [581],
  [joy     ], [95.3%], [—], [695],
  [love    ], [76.7%], [—], [159],
  [anger   ], [91.6%], [—], [275],
  [fear    ], [98.2%], [—], [224],
  [surprise], [57.6%], [—], [66],
)

The confusion matrix shows the same failure structure as Model 2, just
attenuated: *love* is mostly confused with *joy* (35 of 159), and *surprise*
is mostly confused with *fear* (18 of 66) and *joy* (7 of 66). Both pairs are
intuitively close emotions, and *surprise* is the smallest class in the
training set (only 572 training rows, 3.6%), so the model has less signal to
work with regardless of pretraining.

== Analysis of differences vs. the from-scratch models

We ran the same analysis tools (token-level occlusion via a BERT-adapted
variant, and the swap REPL) on the fine-tuned DistilBERT, and observed three
qualitative differences relative to Model 2:

- *Better negation handling.* `i am not happy` is now classified as
  *sadness*, not joy. DistilBERT was pretrained on data that contains plenty
  of negated affect, so the contextual representation of `happy` already
  reflects the preceding `not`. This is the clearest qualitative win over the
  GRU.
- *Less single-token tunnel vision.* Attribution mass is more spread out
  across the sentence. Removing the single most-load-bearing token typically
  drops the predicted class probability by a much smaller amount than it does
  for the GRU, where one token routinely accounts for >60% of the prediction.
- *Same dataset failure modes survive.* The cases that look like dataset
  label noise (the `paradise kissing those sweet lips` joy-vs-love example,
  the `stressed ... are fun` anger-vs-sadness example) are still mistakes.
  Pretraining does not fix label disagreement.

In short: DistilBERT gives us roughly *+2.8 points of test accuracy* over the
best from-scratch Model 2 variant and a clear qualitative improvement on
negation and multi-token reasoning, at the cost of a much larger model (66M
parameters vs. ~1.4M for Model 2) and a dependency on the HuggingFace
ecosystem. For this dataset the headroom over a well-regularised GRU is
modest, which we read as evidence that the remaining error is dominated by
label ambiguity and class imbalance rather than by representation quality.
