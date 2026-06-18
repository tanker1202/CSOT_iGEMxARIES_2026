# Week 3 Assignment — Classical Models for Helix Prediction

[← Week 3 Topics](index.md) | [Leaderboard →](../../leaderboard.md)

---

## Task

Build, tune, and compare **classical machine-learning models** that predict, for every residue in a protein, whether it lies in an **α-helix (1)** or **not (0)** — using only the amino-acid sequence as input. Evaluate your models on a held-out split of proteins and submit a report covering all five deliverables below.

> The training labels are the same DSSP-derived secondary-structure labels you learned to extract in Week 2 — collapsed to a binary target: `H → 1`, `E`/`C → 0`. There is no separate hidden test set this week; setting up an honest held-out evaluation is part of what you are assessed on.

---

## The Dataset

Use the openly available [RCSB Kabsch–Sander secondary-structure dataset](https://www.kaggle.com/datasets/alfrandom/protein-secondary-structure) (a one-click download, no account needed to fetch the file). It is a tabular dump of PDB chains with columns including `pdb_id`, `chain_code`, `seq`, and `sst3` (the per-residue Q3 string over `{H, E, C}`).

- Prefer the **culled / low-redundancy** subset if the download offers one — it keeps your evaluation honest by ensuring train and held-out proteins are dissimilar.
- Derive your binary target from `sst3`: `H → 1`, everything else `→ 0`.
- Non-standard residues may be masked (e.g. `*`); drop those chains or treat the symbol as a 21st token — your call, just document it.

You are free to subsample (e.g. 1,000–2,000 chains) to keep training fast — the task is completable on a laptop in well under a week. If you prefer, you may use any other openly available labelled secondary-structure set (CB513, NetSurfP-2.0, etc.); just say which and why.

### Prep snippet

```python
import pandas as pd

AA = set("ACDEFGHIKLMNPQRSTVWY")
df = pd.read_csv("2018-06-08-ss.cleaned.csv")        # the public CSV

def usable(seq, ss):
    return (isinstance(seq, str) and isinstance(ss, str)
            and len(seq) == len(ss) and 30 <= len(seq) <= 500
            and sum(c not in AA for c in seq) / len(seq) <= 0.10)

df = df[[usable(s, t) for s, t in zip(df.seq, df.sst3)]].reset_index(drop=True)
df["protein_id"] = df.pdb_id.astype(str) + "_" + df.chain_code.astype(str)

rows = []
for _, r in df.iterrows():
    for i, (aa, ss) in enumerate(zip(r.seq, r.sst3)):
        rows.append({"protein_id": r.protein_id, "pos": i, "aa": aa,
                     "seq": r.seq, "y": 1 if ss == "H" else 0})
res = pd.DataFrame(rows)   # one row per residue
```

---

## Deliverables

### 1. Feature Engineering & Comparison

Build **at least two** of the three feature representations from the Week 3 notes and compare them:

- **One-hot sliding window** (your Week 2 encoder; pick and justify a window size).
- **K-mer composition** of the window (try at least one value of `k ∈ {1, 2, 3}`).
- **TF-IDF-weighted k-mers** over the window corpus.

Report, in a table, the **held-out macro-F1** of one fixed model (your choice) trained on each feature set. State which representation you carried forward and why.

### 2. Model Training & Tuning

Train **at least three** models, and your selection **must include at least one probabilistic model** (Naive Bayes *or* HMM) **and at least one discriminative model** (SVM *or* Random Forest / Decision Tree).

For each model, report:

- The hyperparameters you tuned and the search method (e.g. `GridSearchCV`, manual sweep). At minimum, tune the headline knob of each model — `alpha` for Naive Bayes, `C` for the SVM, `max_depth` / `n_estimators` for the forest.
- **Held-out macro-F1, accuracy, precision, and recall** (helix class), obtained with a **group-aware split** (see Deliverable 4).
- A one-line takeaway: what this model got right or wrong relative to the others.

### 3. Confusion Matrix & Error Analysis

For your **best model**, include:

- The 2×2 confusion matrix on your held-out split.
- **2–4 sentences** of error analysis: is the model biased toward the majority class? Where do its mistakes cluster — at helix boundaries (the first/last residues of a run), in short helices, in coil-rich regions? Use what you know about secondary structure to interpret.

### 4. Methodology Statement (anti-leakage)

In **2–3 sentences**, state explicitly how you split train/held-out and confirm that **no protein chain appears in both**. Residues from the same chain are correlated; a per-residue split inflates your numbers and makes them meaningless. Use `GroupShuffleSplit` / `GroupKFold` keyed on `protein_id`.

### 5. Results Summary & Interpretation

Pull it together:

- A results table comparing all your models (macro-F1, accuracy, helix precision/recall) on the held-out split, with your best result highlighted.
- **2–4 sentences** interpreting *why* the winning model and feature set won — and connect it to the biology where you can (do the most useful features correspond to known helix-forming residues like A, E, L, M? do longer k-mers help because helices are periodic?).

---

## Recommended Workflow

```
1. Load and prep the public CSV (snippet above) -> per-residue rows with protein_id

2. Build features
   - implement the window encoder from Week 2 (one-hot)
   - add a k-mer / TF-IDF vectoriser over window strings
   - fit the vectoriser on the TRAINING split only, then transform the held-out
     split (fitting on held-out data = leakage)

3. Group-aware split
   - GroupShuffleSplit(test_size=0.2) on protein_id -> train / held-out indices

4. Train & tune >=3 models (>=1 probabilistic, >=1 discriminative)
   - GridSearchCV with GroupKFold for honest cross-validation

5. Evaluate: macro-F1, accuracy, precision, recall, confusion matrix

6. Write up the report (deliverables 1-5)
```

### Code snippets

**Installation:**

```
pip install pandas numpy scikit-learn hmmlearn
```

**Window encoder + k-mer / TF-IDF features:**

```python
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

def windows(seq, w=17):
    pad = w // 2
    s = "-" * pad + seq + "-" * pad
    return [s[i:i + w] for i in range(len(seq))]

# one window-string per residue, in the same order as `res`
window_strings = []
for seq in res.groupby("protein_id", sort=False)["seq"].first():
    window_strings.extend(windows(seq, w=17))

def kmers(s, k=3):
    return [s[i:i + k] for i in range(len(s) - k + 1)]

vec = TfidfVectorizer(analyzer=lambda s: kmers(s, k=3))   # FIT ON TRAIN ONLY
```

**Group-aware split + tuning + evaluation:**

```python
from sklearn.model_selection import GroupShuffleSplit, GridSearchCV, GroupKFold
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score, classification_report, confusion_matrix

groups = res["protein_id"].values
y = res["y"].values

gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
tr, va = next(gss.split(window_strings, y, groups))
assert set(groups[tr]).isdisjoint(set(groups[va])), "leakage: chain in both splits"

X = vec.fit_transform([window_strings[i] for i in tr])      # fit on train only
Xva = vec.transform([window_strings[i] for i in va])

grid = GridSearchCV(
    LinearSVC(class_weight="balanced"),
    param_grid={"C": [0.01, 0.1, 1, 10]},
    scoring="f1_macro",
    cv=GroupKFold(n_splits=4),
)
grid.fit(X, y[tr], groups=groups[tr])

pred = grid.predict(Xva)
print("held-out macro-F1:", f1_score(y[va], pred, average="macro"))
print(confusion_matrix(y[va], pred))
print(classification_report(y[va], pred, digits=3))
```

**Supervised HMM (count-based, optional model choice):**

```python
import numpy as np
from hmmlearn import hmm

AA = "ACDEFGHIKLMNPQRSTVWY"
idx = {a: i for i, a in enumerate(AA)}

# estimate start / transition / emission tables by counting on the TRAIN split
pi = np.zeros(2); A = np.zeros((2, 2)); B = np.zeros((2, 20))
for seq, ss in train_pairs:                      # (sequence, sst3) for train chains
    labels = [1 if s == "H" else 0 for s in ss]
    pi[labels[0]] += 1
    for i, (aa, lab) in enumerate(zip(seq, labels)):
        if aa in idx: B[lab, idx[aa]] += 1
        if i > 0: A[labels[i - 1], lab] += 1

pi /= pi.sum(); A /= A.sum(1, keepdims=True); B /= B.sum(1, keepdims=True)

model = hmm.CategoricalHMM(n_components=2, init_params="")
model.startprob_, model.transmat_, model.emissionprob_ = pi, A, B

obs = np.array([[idx[a]] for a in some_seq if a in idx])
pred_states = model.predict(obs)                 # Viterbi -> per-residue 0/1
```

---

## Optional (ungraded)

*These earn no points — they are here to deepen your understanding and tee up Week 4.*

- **Beat the boundary problem.** Helix-prediction errors cluster at the *ends* of helices. Add features that encode position-in-run or smooth your predictions with a small post-processing pass, and measure the F1 gain.
- **Peek at Week 4.** Swap your hand-crafted features for ESM2 embeddings (`esm2_t6_8M_UR50D`, from the Week 2 optional section) feeding the *same* SVM/forest. How much does a learned representation buy you over k-mers/TF-IDF? Write 2–3 sentences.

---

## Submission Format

Submit a single PDF or Markdown report (with any code in a `.py` file or notebook attached) containing:

- [ ] Feature-comparison table (≥2 representations, held-out macro-F1 each) + chosen representation justified
- [ ] ≥3 models trained (≥1 probabilistic, ≥1 discriminative) with tuned hyperparameters and metrics
- [ ] Confusion matrix + 2–4 sentence error analysis for the best model
- [ ] Methodology statement confirming a group-aware (no chain in both splits) evaluation
- [ ] Results summary table + 2–4 sentence interpretation tying the winner to the biology
- [ ] Code used (`.py` file or notebook)

**Deadline**: Jun 21, 2026 — results posted to the [leaderboard](../../leaderboard.md) within 48 hours.

---

## Marking Criteria

| Component | Weight | What we look for |
|-----------|--------|-----------------|
| **Correctness** | 60% | Sound feature engineering, correctly trained and tuned models, a genuinely leak-free group-aware evaluation, accurate metrics and confusion matrix |
| **Efficiency** | 20% | Clean, non-repetitive code; sensible use of sparse features and vectorised operations; appropriate model/feature choices; no needless brute force |
| **Clarity and Explanation** | 20% | Well-structured report, clear model-comparison tables, correct reading of the confusion matrix, thoughtful interpretation linking results to secondary-structure biology |

### Correctness breakdown (60 pts)

| Sub-component | Points |
|---------------|--------|
| Feature representations built correctly; comparison is valid | 10 |
| ≥3 models (≥1 probabilistic + ≥1 discriminative), correctly trained and tuned | 20 |
| Group-aware split with no chain leakage (correctly implemented and verified) | 15 |
| Metrics, confusion matrix, and best-model results are accurate and sound | 15 |

---

[← Week 3 Topics](index.md) | [Leaderboard →](../../leaderboard.md)
