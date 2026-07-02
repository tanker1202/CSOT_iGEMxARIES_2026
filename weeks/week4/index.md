# Week 4 — Deep Learning for Sequences

[← Home](../../README.md) | [Week 4 Assignment →](assignment.md)

---

> **Building on Week 3.** You beat a majority baseline with classical models on *hand-crafted* features — one-hot windows, k-mers, TF-IDF. This week you stop hand-crafting and let the model **learn the features itself**. You will meet the three architectures that define modern sequence modelling — CNNs, LSTMs, and Transformers — build them in PyTorch, and use them to predict the full 3-class secondary structure (H / E / C) directly from sequence.

---

## Topics

1. [PyTorch for Sequence Data](#1-pytorch-for-sequence-data)
2. [1D CNNs & Motif Discovery](#2-1d-cnns--motif-discovery)
3. [RNNs & LSTMs for Context](#3-rnns--lstms-for-context)
4. [Transformers & Self-Attention](#4-transformers--self-attention)
5. [The Week 4 Task — Q3 Prediction with Deep Nets](#5-the-week-4-task)

---

## 1. PyTorch for Sequence Data

### Why a new toolkit?

Classical models took a fixed-length feature vector and returned a label. Deep networks instead consume the **raw sequence** and learn their own internal representation end-to-end, adjusting millions of parameters by gradient descent. [PyTorch](https://pytorch.org/) is the library we use to define those networks, run them on a GPU, and differentiate them automatically.

The two ideas you need from PyTorch:

- A **tensor** is an n-dimensional array (like NumPy) that can live on a GPU and remembers the operations applied to it.
- **Autograd** records those operations so that `loss.backward()` computes every gradient for you — you never hand-derive a derivative.

→ [PyTorch 60-minute blitz](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html)

### The problem unique to sequences: variable length

A batch of proteins have different lengths, but a tensor must be rectangular. The fix is **padding**: pad every sequence in a batch up to the longest one, and carry a **mask** that marks which positions are real. Everything downstream — the loss, the accuracy — must then **ignore the padded positions**, or your numbers will be silently wrong.

| Concept | What it does |
|---------|-------------|
| **`Dataset`** | Turns one protein into `(input_tensor, label_tensor)` |
| **`DataLoader`** | Batches, shuffles, and parallelises loading |
| **`collate_fn`** | Custom batching — here, pads the batch and builds the mask |
| **padding + mask** | Makes the batch rectangular *and* records which positions are real |

### Representing a residue as input

You have three options, in increasing power:

| Input | Dimensionality | Comes from |
|-------|---------------|-----------|
| **Integer index + `nn.Embedding`** | learned, e.g. 32–128 | a lookup table trained with the model |
| **One-hot window** | 20 (or `20 × window`) | Week 2 |
| **ESM2 / ProtT5 embeddings** | 320–1024 | Week 2 optional — a pretrained Transformer |

An `nn.Embedding` is the deep-learning analogue of one-hot: instead of a fixed 20-dim binary vector, each amino acid gets a **dense, learned** vector that the network is free to shape however helps the task.

### The data pipeline (you will reuse this all week)

```python
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

AA = "ACDEFGHIKLMNPQRSTVWY"
aa_idx = {a: i for i, a in enumerate(AA)}
PAD_IDX = 20                      # a 21st token used only for padding
Q3 = {"H": 0, "E": 1, "C": 2}
IGNORE = -100                     # CrossEntropyLoss skips this label

class SSData(Dataset):
    def __init__(self, df):       # df has columns: seq, sst3
        self.rows = df.reset_index(drop=True)
    def __len__(self):
        return len(self.rows)
    def __getitem__(self, i):
        r = self.rows.iloc[i]
        x = torch.tensor([aa_idx.get(c, PAD_IDX) for c in r.seq])
        y = torch.tensor([Q3.get(c, IGNORE) for c in r.sst3])
        return x, y

def collate(batch):
    xs, ys = zip(*batch)
    X = pad_sequence(xs, batch_first=True, padding_value=PAD_IDX)
    Y = pad_sequence(ys, batch_first=True, padding_value=IGNORE)
    mask = (X != PAD_IDX)         # True at real residues, False at padding
    return X, Y, mask
```

> **The masking rule, stated once:** padded positions must never contribute to the loss or the accuracy. We achieve this by labelling them `IGNORE = -100` (which `nn.CrossEntropyLoss(ignore_index=-100)` skips) and by filtering on the mask when we score. Forget this and you will "achieve" 95% accuracy by correctly predicting padding.

→ [PyTorch Dataset & DataLoader](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html)
→ [`pad_sequence` docs](https://pytorch.org/docs/stable/generated/torch.nn.utils.rnn.pad_sequence.html)

---

## 2. 1D CNNs & Motif Discovery

### From fixed windows to learned motifs

In Week 2 you slid a fixed window along the sequence; in Week 3 you counted k-mers. A **[1D convolution](https://en.wikipedia.org/wiki/Convolutional_neural_network)** does something that subsumes both: it slides a small **learned filter** along the sequence and fires when it sees the pattern it has learned to detect. Each filter is, in effect, a **motif detector** — and unlike a k-mer table, the network *discovers* which motifs matter for the task.

```
sequence embeddings:  [e1 e2 e3 e4 e5 e6 e7 ...]
filter (width 3):        └─┬─┘
                       slides across, output high where the motif matches
```

Stack a few convolutional layers and the **receptive field** grows: layer 1 sees 3–7 residues, layer 2 sees a wider span, and so on — the network builds up from local motifs to larger structural context.

### The one shape gotcha

`nn.Conv1d` expects `(batch, channels, length)`, but our embeddings come out as `(batch, length, channels)`. So you **transpose** in and out. Use `padding = kernel_size // 2` to keep the length unchanged (so you still get one prediction per residue).

```python
N_CLASSES = 3     # H / E / C

class CNN(nn.Module):
    def __init__(self, d=32):
        super().__init__()
        self.emb = nn.Embedding(21, d, padding_idx=PAD_IDX)
        self.body = nn.Sequential(
            nn.Conv1d(d, 64, kernel_size=7, padding=3), nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=7, padding=3), nn.ReLU(),
        )
        self.head = nn.Conv1d(64, N_CLASSES, kernel_size=1)   # per-residue logits
    def forward(self, x):                 # x: (B, L)
        h = self.emb(x).transpose(1, 2)   # (B, d, L)
        h = self.body(h)                  # (B, 64, L)
        return self.head(h).transpose(1, 2)   # (B, L, C)
```

CNNs are **fast**, **parallel** across positions, and excellent at exactly the kind of local pattern (a run of helix-favouring residues) that drives secondary structure. Their weakness is long-range context — a filter only sees its receptive field. That is what the next two architectures fix.

> **Interpretability bonus.** After training, a filter's weights can be read like a little position weight matrix — inspect which amino acids excite each filter and you can often recognise learned helix/strand motifs. A nice thing to include in your report.

→ [PyTorch `Conv1d` docs](https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html)
→ [CS231n: Convolutional Networks](https://cs231n.github.io/convolutional-networks/)

---

## 3. RNNs & LSTMs for Context

### The idea: carry a memory along the chain

A **[recurrent neural network](https://en.wikipedia.org/wiki/Recurrent_neural_network)** walks along the sequence one residue at a time, updating a **hidden state** that summarises everything seen so far. That hidden state is a memory — in principle it lets a prediction at position `i` depend on residues far to the left.

In practice, plain RNNs **forget**: gradients shrink as they propagate back through many steps (the [vanishing gradient problem](https://en.wikipedia.org/wiki/Vanishing_gradient_problem)), so long-range signal is lost.

### LSTMs fix the memory

A **[Long Short-Term Memory](https://en.wikipedia.org/wiki/Long_short-term_memory)** cell adds a protected **cell state** and three **gates** — forget, input, output — that decide what to keep, add, and expose at each step. This lets useful information flow across long distances without vanishing.

### Why *bi*directional is non-negotiable here

Whether a residue is in a helix depends on its neighbours on **both** sides. A one-directional LSTM only sees the left context. A **bidirectional LSTM (BiLSTM)** runs one LSTM left-to-right and another right-to-left and concatenates them, so every position's representation is informed by the entire sequence. For per-residue secondary-structure prediction, BiLSTM is the natural classical-deep choice.

```python
class BiLSTM(nn.Module):
    def __init__(self, d=32, hidden=64, layers=2):
        super().__init__()
        self.emb = nn.Embedding(21, d, padding_idx=PAD_IDX)
        self.lstm = nn.LSTM(d, hidden, num_layers=layers,
                            batch_first=True, bidirectional=True, dropout=0.2)
        self.head = nn.Linear(2 * hidden, N_CLASSES)   # 2x for both directions
    def forward(self, x):                 # x: (B, L)
        h, _ = self.lstm(self.emb(x))     # (B, L, 2*hidden)
        return self.head(h)               # (B, L, C)
```

> **Efficiency note.** For speed you can wrap inputs with `pack_padded_sequence` so the LSTM skips padding. It is optional for this assignment — masking the loss is enough for correctness — but worth knowing.

→ [StatQuest: LSTM with PyTorch](https://www.youtube.com/watch?v=RHGiXPuo_pI)
→ [colah: Understanding LSTMs](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
→ [Stanford CS224n](http://web.stanford.edu/class/cs224n/) — sequence-model lectures and notes

---

## 4. Transformers & Self-Attention

### The limitation attention removes

An RNN is **sequential**: it must process residue 1 before residue 2, which is slow and still strains over very long ranges. The **[Transformer](https://en.wikipedia.org/wiki/Transformer_(deep_learning_architecture))** throws out recurrence entirely and lets every residue look at **every other residue at once**.

### Self-attention in one picture

For each residue, self-attention asks: *"which other residues should I pay attention to when deciding my own representation?"* It answers by comparing a **query** for that residue against a **key** from every residue, turning the comparisons into weights, and taking a weighted sum of every residue's **value**:

```
Attention(Q, K, V) = softmax( Q Kᵀ / √d ) V
```

- **Q, K, V** are three learned linear projections of the input.
- `Q Kᵀ` scores how relevant every residue is to every other — an `L × L` table.
- softmax turns each row into weights; the output mixes the values accordingly.

**Multi-head** attention runs several of these in parallel so the model can attend to different kinds of relationship at once. Because attention is order-blind, we add a **positional encoding** so the model knows *where* each residue sits.

### The encoder — what you will build

Stack `N` blocks of (multi-head self-attention → feed-forward), each with residual connections and layer norm. Because a residue's structure depends on the whole chain, an **encoder** (which sees the full sequence bidirectionally) is what we want — the same design as the protein language models from Week 2.

```python
class PosEnc(nn.Module):
    def __init__(self, d, maxlen=2000):
        super().__init__()
        pe = torch.zeros(maxlen, d); pos = torch.arange(maxlen).unsqueeze(1)
        div = torch.exp(torch.arange(0, d, 2) * (-torch.log(torch.tensor(10000.0)) / d))
        pe[:, 0::2] = torch.sin(pos * div); pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)
    def forward(self, x):
        return x + self.pe[:x.size(1)]

class TransformerSS(nn.Module):
    def __init__(self, d=64, nhead=4, layers=2):
        super().__init__()
        self.emb = nn.Embedding(21, d, padding_idx=PAD_IDX)
        self.pos = PosEnc(d)
        layer = nn.TransformerEncoderLayer(d, nhead, dim_feedforward=128,
                                           batch_first=True)
        self.enc = nn.TransformerEncoder(layer, layers)
        self.head = nn.Linear(d, N_CLASSES)
    def forward(self, x, mask):           # mask: True at real residues
        h = self.pos(self.emb(x))
        h = self.enc(h, src_key_padding_mask=~mask)   # NB: True = "ignore this position"
        return self.head(h)               # (B, L, C)
```

> **The mask trap.** `src_key_padding_mask` uses the *opposite* convention to ours: `True` means *ignore*. So you pass `~mask`. Getting this backwards makes the model attend only to padding — a silent, score-wrecking bug.

### The connection back to Week 2

ESM2 and ProtT5 *are* Transformer encoders — just pretrained on hundreds of millions of sequences. That is exactly why their embeddings (Week 2 optional) carry so much structure. This week you can either train a small Transformer from scratch, or feed **ESM2 embeddings** into any of these heads and watch accuracy jump — a preview of why pretraining dominates modern biology.

→ [Attention Is All You Need — Vaswani et al. (2017)](https://arxiv.org/pdf/1706.03762.pdf)
→ [3Blue1Brown: Neural Networks & Attention](https://youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)
→ [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
→ [PyTorch `TransformerEncoderLayer` docs](https://pytorch.org/docs/stable/generated/torch.nn.TransformerEncoderLayer.html)

### The three architectures at a glance

| Model | Sees context | Speed | Best at |
|-------|-------------|-------|---------|
| **1D CNN** | local (receptive field) | fastest | local motifs |
| **BiLSTM** | full sequence, sequentially | medium | ordered context, medium range |
| **Transformer** | full sequence, all-at-once | fast on GPU, memory-hungry | long-range dependencies |

---

## 5. The Week 4 Task

### The prediction problem

> Given a protein's amino-acid sequence, predict the **Q3 secondary-structure class — H, E, or C — for every residue.**

This upgrades Week 3's binary helix/non-helix to the **full 3-class** problem, and you solve it with networks that learn their own features. Optionally, predict the finer **Q8** classes (the ungraded extension).

### The metric: Q3 accuracy

**Q3 accuracy** is the fraction of residues whose predicted class equals the true class — computed **only over real (non-padded) residues**:

```python
def q3_accuracy(logits, y):           # logits: (B, L, C), y: (B, L)
    pred = logits.argmax(-1)
    keep = (y != IGNORE)              # drop padded positions
    return (pred[keep] == y[keep]).float().mean().item()
```

Report per-class performance too (a confusion matrix over H/E/C) — coil is the majority class, and a model can post a decent Q3 while barely predicting strands.

### The two disciplines that decide whether your number is real

1. **Group-aware split — by protein chain.** Because a deep model consumes a whole chain at once, the natural unit of splitting is the chain: put each chain entirely in train *or* held-out, never split a chain's residues across both. Use the low-redundancy dataset so held-out chains are truly unseen.
2. **Mask the padding — everywhere.** Padded positions must be excluded from the loss (`ignore_index=-100`) *and* the accuracy (`keep = y != IGNORE`). This is the single most common way Week 4 reports come out wrong.

### Beat your Week 3 baseline

You already have a classical macro-F1 number from last week. A well-trained BiLSTM or Transformer should clearly beat it — showing the difference between hand-crafted and learned representations. Make that comparison explicit in your report.

### The pipeline

```
labelled dataset (sequence + H/E/C)
        |
        v
  SSData + DataLoader(collate)  -->  pad batch + build mask
        |
        v
  input:  learned nn.Embedding  |  one-hot (Wk2)  |  ESM2 embeddings (Wk2)
        |
        v
  model:  1D CNN   AND   ( BiLSTM  OR  Transformer encoder )
        |
        |  train with masked CrossEntropyLoss; split by chain
        v
  evaluate on held-out chains  -->  Q3 accuracy, training curves, H/E/C confusion matrix
                                    (compare against the Week 3 classical baseline)
```

> **Looking ahead.** Next week pulls the whole course together — combining the representations, models, and evaluation discipline you have built across Weeks 1–5 into a final secondary-structure predictor.

---

## Resources

- [PyTorch tutorials](https://pytorch.org/tutorials/) — Dataset/DataLoader, training loops, RNNs, Transformers
- [StatQuest: LSTM with PyTorch](https://www.youtube.com/watch?v=RHGiXPuo_pI) — gentle LSTM walkthrough
- [3Blue1Brown: Neural Networks & Attention](https://youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi) — the best visual intuition for attention
- [Stanford CS224n](http://web.stanford.edu/class/cs224n/) — sequence models, RNNs, attention (notes and lectures)
- [Attention Is All You Need — Vaswani et al. (2017)](https://arxiv.org/pdf/1706.03762.pdf) — the Transformer paper
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) — annotated walkthrough of the paper
- [colah: Understanding LSTMs](https://colah.github.io/posts/2015-08-Understanding-LSTMs/) — the classic LSTM explainer
- [ESM2 repository](https://github.com/facebookresearch/esm) — pretrained Transformer embeddings you can feed to any head
- [Protein Secondary Structure dataset (RCSB Kabsch–Sander, curated CSV)](https://www.kaggle.com/datasets/alfrandom/protein-secondary-structure) — the labelled dataset used for this week's task

---

[← Home](../../README.md) | [Week 4 Assignment →](assignment.md)
