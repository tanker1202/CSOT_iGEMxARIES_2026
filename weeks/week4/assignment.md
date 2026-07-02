# Week 4 Assignment — Deep Models for Secondary-Structure Prediction

[← Week 4 Topics](index.md) | [Leaderboard →](../../leaderboard.md)

---

## Task

Implement **deep-learning architectures** that predict the **Q3 secondary-structure class (H / E / C) for every residue** of a protein, from sequence alone. Train them properly, evaluate on a held-out split of proteins, and submit a report comparing your models — including against your Week 3 classical baseline.

> This upgrades last week's binary helix/non-helix to the full 3-class problem, solved with networks that learn their own features. The labels are the DSSP-derived Q3 strings from Week 2; the metric is **Q3 accuracy** computed over real (non-padded) residues.

---

## The Dataset

Use the same openly available [RCSB Kabsch–Sander secondary-structure dataset](https://www.kaggle.com/datasets/alfrandom/protein-secondary-structure) as Week 3 (columns `pdb_id`, `chain_code`, `seq`, `sst3`; `sst8` is also present if you attempt the Q8 extension).

- Map each residue's `sst3` character to a class index: `H → 0`, `E → 1`, `C → 2`.
- Prefer the **culled / low-redundancy** subset so held-out chains are genuinely unseen.
- Subsample to a few thousand chains if you want faster epochs — deep models train fine on a laptop CPU at this scale, and much faster on any GPU (Colab is plenty).
- You may instead use any standard labelled set (CB513, NetSurfP-2.0, etc.) — just say which.

The prep is the same shape as Week 3; the only change is a 3-class target instead of binary. Split **by chain** into train / held-out (each chain wholly in one split).

---

## Deliverables

### 1. Data Pipeline

Build a PyTorch input pipeline and describe it briefly:

- A `Dataset` that turns each chain into `(input_tensor, label_tensor)`.
- A `DataLoader` with a `collate_fn` that **pads** each batch and builds a **mask**.
- State your input representation and justify it: learned `nn.Embedding`, one-hot windows (Week 2), or ESM2 embeddings (Week 2 optional).
- Confirm in one line how padded positions are excluded from the loss (e.g. `ignore_index=-100`).

### 2. CNN Model

Implement a **1D CNN** for per-residue Q3 prediction and report:

- The architecture (layers, kernel sizes, channels) and training setup (optimiser, learning rate, epochs, batch size).
- A **training curve** (loss and/or Q3 accuracy vs epoch) for train and held-out.
- Final **held-out Q3 accuracy**.

### 3. A Context Model — BiLSTM *or* Transformer Encoder

Implement **one** sequence-context architecture — a **BiLSTM** or a **Transformer encoder** — and report the same three items (architecture + training setup, training curve, held-out Q3 accuracy).

> If you build the Transformer, get the padding mask right: `src_key_padding_mask` uses `True = ignore`, so pass `~mask`. If you build the BiLSTM, make it **bidirectional** — a residue's structure depends on both neighbours.

### 4. Comparison & Error Analysis

Bring the models together:

- A results table: **held-out Q3 accuracy** for your CNN, your context model, and your **Week 3 classical baseline** (quote last week's number, or re-run it).
- A **confusion matrix over H / E / C** for your best model, plus **per-class accuracy**.
- **3–5 sentences**: which architecture won and why? Where do the errors concentrate — strands (usually the hardest class), helix/coil boundaries, short segments? Did the deep models beat the classical baseline, and by how much?

### 5. Methodology Statement

In **2–3 sentences**, confirm the two disciplines that make your numbers trustworthy:

- **Group-aware split by chain** — no chain appears in both train and held-out.
- **Masking** — padded positions are excluded from both the loss *and* the Q3-accuracy computation.

---

## Recommended Workflow

```
1. Load & prep the dataset -> DataFrame of chains (seq, sst3); split BY CHAIN

2. Build the pipeline: SSData + DataLoader(collate) -> padded batches + mask

3. Model A: 1D CNN  (fast to train; get the end-to-end loop working here first)

4. Model B: BiLSTM or Transformer encoder

5. Train each: masked CrossEntropyLoss, Adam, a handful of epochs;
   track train/held-out Q3 accuracy per epoch

6. Evaluate: held-out Q3 accuracy, confusion matrix, per-class accuracy

7. Compare to the Week 3 classical baseline; write up (deliverables 1-5)
```

### Code snippets

**Installation:**

```
pip install torch pandas numpy scikit-learn
```

**Data pipeline (padding + mask):**

```python
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

AA = "ACDEFGHIKLMNPQRSTVWY"
aa_idx = {a: i for i, a in enumerate(AA)}
PAD_IDX = 20; Q3 = {"H": 0, "E": 1, "C": 2}; IGNORE = -100; N_CLASSES = 3

class SSData(Dataset):
    def __init__(self, df): self.rows = df.reset_index(drop=True)
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        r = self.rows.iloc[i]
        x = torch.tensor([aa_idx.get(c, PAD_IDX) for c in r.seq])
        y = torch.tensor([Q3.get(c, IGNORE) for c in r.sst3])
        return x, y

def collate(batch):
    xs, ys = zip(*batch)
    X = pad_sequence(xs, batch_first=True, padding_value=PAD_IDX)
    Y = pad_sequence(ys, batch_first=True, padding_value=IGNORE)
    return X, Y, (X != PAD_IDX)          # mask: True at real residues
```

**Models (CNN, BiLSTM, Transformer):**

```python
class CNN(nn.Module):
    def __init__(self, d=32):
        super().__init__()
        self.emb = nn.Embedding(21, d, padding_idx=PAD_IDX)
        self.body = nn.Sequential(
            nn.Conv1d(d, 64, 7, padding=3), nn.ReLU(),
            nn.Conv1d(64, 64, 7, padding=3), nn.ReLU())
        self.head = nn.Conv1d(64, N_CLASSES, 1)
    def forward(self, x):
        h = self.emb(x).transpose(1, 2)
        return self.head(self.body(h)).transpose(1, 2)      # (B, L, C)

class BiLSTM(nn.Module):
    def __init__(self, d=32, hidden=64, layers=2):
        super().__init__()
        self.emb = nn.Embedding(21, d, padding_idx=PAD_IDX)
        self.lstm = nn.LSTM(d, hidden, layers, batch_first=True,
                            bidirectional=True, dropout=0.2)
        self.head = nn.Linear(2 * hidden, N_CLASSES)
    def forward(self, x):
        return self.head(self.lstm(self.emb(x))[0])          # (B, L, C)

class PosEnc(nn.Module):
    def __init__(self, d, maxlen=2000):
        super().__init__()
        pe = torch.zeros(maxlen, d); pos = torch.arange(maxlen).unsqueeze(1)
        div = torch.exp(torch.arange(0, d, 2) * (-torch.log(torch.tensor(10000.0)) / d))
        pe[:, 0::2] = torch.sin(pos * div); pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)
    def forward(self, x): return x + self.pe[:x.size(1)]

class TransformerSS(nn.Module):
    def __init__(self, d=64, nhead=4, layers=2):
        super().__init__()
        self.emb = nn.Embedding(21, d, padding_idx=PAD_IDX)
        self.pos = PosEnc(d)
        layer = nn.TransformerEncoderLayer(d, nhead, 128, batch_first=True)
        self.enc = nn.TransformerEncoder(layer, layers)
        self.head = nn.Linear(d, N_CLASSES)
    def forward(self, x, mask):
        h = self.enc(self.pos(self.emb(x)), src_key_padding_mask=~mask)
        return self.head(h)                                  # (B, L, C)
```

**Training loop + masked Q3 accuracy:**

```python
def q3_accuracy(logits, y):
    pred = logits.argmax(-1); keep = (y != IGNORE)
    return (pred[keep] == y[keep]).float().mean().item()

device = "cuda" if torch.cuda.is_available() else "cpu"
model = CNN().to(device)                       # swap in BiLSTM() / TransformerSS()
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss(ignore_index=IGNORE)

for epoch in range(20):
    model.train()
    for X, Y, mask in train_loader:
        X, Y, mask = X.to(device), Y.to(device), mask.to(device)
        opt.zero_grad()
        logits = model(X, mask) if isinstance(model, TransformerSS) else model(X)
        loss = loss_fn(logits.transpose(1, 2), Y)   # CE wants (B, C, L) vs (B, L)
        loss.backward(); opt.step()

    model.eval(); accs = []
    with torch.no_grad():
        for X, Y, mask in val_loader:
            X, Y, mask = X.to(device), Y.to(device), mask.to(device)
            logits = model(X, mask) if isinstance(model, TransformerSS) else model(X)
            accs.append(q3_accuracy(logits, Y))
    print(f"epoch {epoch}: held-out Q3 acc = {sum(accs)/len(accs):.3f}")
```

---

## Optional (ungraded)

*No points — for depth and to preview where the field goes.*

- **Q8 prediction.** Predict the finer 8-class DSSP label (`sst8`) and report Q8 accuracy alongside Q3.
- **Pretrained embeddings.** Feed **ESM2** per-residue embeddings (Week 2 optional) into any head instead of a learned `nn.Embedding`, and measure the jump.
- **Attention map.** For the Transformer, visualise an attention head on one protein — does it attend along helices/strands?
- **Class weighting.** Strands (E) are the minority and hardest class; try `CrossEntropyLoss(weight=...)` and report the effect on per-class accuracy.

---

## Submission Format

Submit a single PDF or Markdown report (with code as a `.py` file or notebook attached) containing:

- [ ] Data-pipeline description + input representation justified + masking confirmed
- [ ] CNN: architecture, training setup, training curve, held-out Q3 accuracy
- [ ] Context model (BiLSTM or Transformer): architecture, training curve, held-out Q3 accuracy
- [ ] Comparison table (CNN vs context model vs Week 3 baseline) + H/E/C confusion matrix + per-class accuracy
- [ ] 3–5 sentence analysis of which model won and where the errors are
- [ ] Methodology statement (group-aware split by chain + masking of padding)
- [ ] Code (`.py` file or notebook)

**Deadline**: Jun 28, 2026 — results posted to the [leaderboard](../../leaderboard.md) within 48 hours.

---

## Marking Criteria

| Component | Weight | What we look for |
|-----------|--------|-----------------|
| **Correctness** | 60% | A working padded/masked pipeline, correctly implemented CNN and context model, a genuinely leak-free chain-level split, accurate Q3 accuracy / confusion matrix, and a fair comparison to the classical baseline |
| **Efficiency** | 20% | Sensible model sizes and batch/epoch choices; use of GPU/batching where available; masking rather than brute force; no needless retraining or dead code |
| **Clarity and Explanation** | 20% | Well-structured report, readable training curves, correct reading of the confusion matrix, and a clear narrative of why one architecture beat the others |

### Correctness breakdown (60 pts)

| Sub-component | Points |
|---------------|--------|
| Data pipeline: padding + mask + label handling correct | 10 |
| CNN and one context model correctly implemented and trained | 20 |
| Chain-level split (no leakage) **and** padding masked in loss *and* metric | 15 |
| Q3 accuracy, curves, and confusion matrix accurate; baseline comparison sound | 15 |

---

[← Week 4 Topics](index.md) | [Leaderboard →](../../leaderboard.md)
