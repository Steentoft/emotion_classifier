= Model 2
For our second model we moved from attention-based architectures to a recurrent one, based on one of the exercise sessions. We started with a vanilla RNN, but quickly ran into vanishing gradients over long sequences, which matters here because many tweets in the dataset are long and the emotional cue can sit at either end of the sentence. To address this we switched to a Gated Recurrent Unit (GRU), which uses update and reset gates to control how much past information is carried forward and largely mitigates the vanishing-gradient problem.

- *Why Bidirectional?*
A unidirectional GRU only conditions each hidden state on the past. For emotion classification that is a problem, since the polarity of a sentence often depends on tokens that appear *later* in the sequence. An example from the training set:

"i am not amazing or great at photography *but* i feel passionate about it"

Read left-to-right, the first half of the sentence strongly suggests *sadness*. Only after the word "but" does the actual emotion (*joy*) become clear. A Bidirectional GRU runs two GRUs in parallel, one forward and one backward, so the representation of every token has access to both past and future context. We concatenate the final forward and backward hidden states of the top layer and feed the resulting vector into a linear classifier.

```python
class TextBiGRU(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes,
                 pad_idx=PAD_ID, dropout=0.3, num_layers=2,
                 pretrained_embeddings=None, freeze_embeddings=False):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.emb_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(
            embed_dim, hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.out_dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
```

- *Handling Variable-Length Input*
Tweets in the dataset vary a lot in length, so running the GRU over padded batches would waste compute and, worse, let `<PAD>` tokens leak into the final hidden state. We use `pack_padded_sequence` so the GRU only processes real tokens, and then take the hidden state from the last non-pad position of each direction:

```python
lengths = (input_ids != self.pad_idx).sum(dim=1).clamp(min=1)
packed = nn.utils.rnn.pack_padded_sequence(
    emb, lengths.cpu(), batch_first=True, enforce_sorted=False
)
_, hidden = self.gru(packed)
forward_hidden = hidden[-2]
backward_hidden = hidden[-1]
hidden = torch.cat((forward_hidden, backward_hidden), dim=1)
```

- *Training and Hyperparameters*
We trained a 2-layer Bidirectional GRU with a hidden size of 256. Optimization used AdamW with a learning rate of 1e-3 and weight decay of 1e-2, together with Cross-Entropy Loss. To keep training stable we applied gradient clipping at a maximum norm of 1.0, and dropout of 0.3 on the embeddings, between GRU layers, and before the classifier.

== Attempts to optimize the model
The baseline Bidirectional GRU already beat a plain RNN, but we still needed several extensions to push macro F1 higher and to fight overfitting on a relatively small training set.

- *Why Macro F1?*
Accuracy is misleading on this dataset because the label distribution is heavily skewed: *joy* and *sadness* together cover most of the training examples, while *love* and *surprise* are rare. A model that always predicts the two majority classes can score very high on raw accuracy while completely failing on the minority classes. So we optimized and reported *macro F1*, the unweighted mean of the per-class F1 scores. Macro F1 weights every emotion equally regardless of frequency, so an improvement on *surprise* counts as much as one on *joy*. We use macro F1 as the criterion for both the best-checkpoint selection and the early-stopping signal:

```python
from sklearn.metrics import f1_score
macro_f1 = f1_score(all_targets, all_preds, average="macro")

if macro_f1 > best_f1:
    best_f1 = macro_f1
    best_state = {k: v.detach().cpu().clone()
                  for k, v in model.state_dict().items()}
    epochs_no_improve = 0
else:
    epochs_no_improve += 1
```

- *Class-Weighted Loss*
With the imbalance above, training under an unweighted loss rewards the model for ignoring the minority classes: accuracy stays high while macro F1 collapses. We computed inverse-frequency class weights from the training set and passed them to `CrossEntropyLoss`:

```python
counts = torch.bincount(data["trainY"], minlength=num_classes).float()
class_weights = counts.sum() / (num_classes * counts.clamp(min=1))
loss_fn = nn.CrossEntropyLoss(weight=class_weights)
```

This improved macro F1 noticeably even when overall accuracy moved only slightly.

- *LR Scheduling and Early Stopping*
We attached a `ReduceLROnPlateau` scheduler that halves the learning rate whenever validation loss stalls, and added an early-stopping criterion that terminates training after 5 epochs without an improvement in macro F1. We always restore the best checkpoint (by validation macro F1) before evaluation rather than the final epoch, so any overfitting late in training does not affect the reported numbers.

- *Pretrained GloVe Embeddings*
With only ~16k training rows, learning a full embedding table from scratch leaves many tokens with essentially random vectors. We therefore initialized the embedding layer with pretrained GloVe 6B 100-dimensional vectors. Each tiktoken-derived vocabulary entry is decoded back to a string and looked up in the GloVe table; tokens with no match keep their random initialization. We tried two variants:

  - *Frozen embeddings*: GloVe vectors are held fixed and only the GRU and classifier are trained.
  - *Fine-tuned embeddings*: GloVe vectors are used as initialization but updated during training.

The actual impact on the test set was smaller than we expected:

#table(
  columns: 3,
  [*Embedding init*], [*Test accuracy*], [*Macro F1*],
  [From scratch], [0.8925], [0.8484],
  [GloVe frozen], [0.8915], [0.8513],
  [GloVe fine-tuned], [0.9015], [0.8630],
)

The frozen variant is essentially identical to the from-scratch baseline, and fine-tuning gives only a modest improvement (+1.5 points macro F1, +0.9 points accuracy). A few reasons explain why pretrained embeddings help less here than they usually do on small text-classification tasks:

  1. *Tokenizer mismatch*. We tokenize with tiktoken BPE, but GloVe is keyed on whitespace-separated words. Many tiktoken pieces are sub-word fragments (e.g. "ing", "feel", "passionat") that either do not appear in the GloVe vocabulary at all, or whose GloVe vector does not represent the same unit of meaning. So a substantial portion of the embedding table still falls back to random initialization and GloVe gives a smaller head-start than it would on a word-level tokenizer.
  2. *Domain mismatch*. GloVe 6B is trained on Wikipedia and Gigaword, which are formal, encyclopedic text. Our dataset is short, informal, first-person, tweet-style sentences dominated by emotion vocabulary. The geometry GloVe captures (topical similarity between common nouns and named entities) does not line up well with what the classifier needs: affective polarity, intensity, and the role of negation and discourse markers like "but".
  3. *The task is mostly local*. Many examples are decided by a handful of high-signal tokens ("love", "scared", "furious", "i don't feel"). Once class-weighted loss and augmentation push the model to attend to those tokens, the embedding layer learns serviceable representations from the 16k training rows on its own. The bottleneck is the GRU + classifier capacity, not the embedding quality.
  4. *Other regularizers have a bigger effect*. Class-weighted CrossEntropy, prefix stripping and word dropout each move macro F1 more than pretrained embeddings do. Once those are in place, the added value of better-initialized embeddings shrinks.

Fine-tuning still slightly outperformed the frozen variant, which fits the analysis above: letting the embeddings drift gives the model a way to partially correct the domain mismatch on the words it does have GloVe vectors for. So we kept the fine-tuned variant in the final configuration, but we would not call GloVe a major contributor to this model's performance.

- *Data Augmentation*
Looking at the dataset revealed two patterns the model was prone to memorize, so we added two on-the-fly augmentations applied to each training batch:

  1. *Prefix stripping*: a large fraction of the sentences start with templated phrases like "i feel", "i am feeling", "ive been feeling", "i felt". With probability `p_strip = 0.4` we strip the longest matching prefix from a row, forcing the model to classify the remaining content rather than lean on the prompt template as a shortcut.
  2. *Word dropout*: with probability `p_word_drop = 0.1` each non-pad token in a batch is replaced by `<UNK>`. This acts as a regularizer at the input level and stops the model from leaning too heavily on any single keyword.

Both augmentations are bounded by a `min_keep_tokens` constraint so we never reduce a sample to an empty sequence.

- *Final Configuration*
The best-performing configuration combined all of the above: a 2-layer Bidirectional GRU with hidden size 256, GloVe-initialized fine-tuned embeddings, dropout of 0.3, AdamW with weight decay 1e-2, class-weighted Cross-Entropy, gradient clipping at norm 1.0, `ReduceLROnPlateau` scheduling, early stopping on macro F1, and the two augmentations above.
