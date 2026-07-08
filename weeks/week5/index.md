# Week 5 — Evolutionary Features, PSSM Profiles & Hyperparameter Tuning

[← Home](../../README.md)

---

> **Building on Weeks 3 & 4.** You now have a working Q3 prediction pipeline: classical models trained on one-hot / k-mer features, and deep networks (CNN, BiLSTM, Transformer) trained on learned embeddings. Both approaches use only the raw sequence of a single protein. This week you add a second, much richer source of signal — **evolution itself** — and then apply systematic **hyperparameter tuning** to squeeze the best possible Q3 accuracy out of every model before the final pipeline in Week 6.

---

## Topics

1. [Why Evolution Helps — The Logic of Evolutionary Features](#1-why-evolution-helps)
2. [Multiple Sequence Alignment as a Profile](#2-multiple-sequence-alignment-as-a-profile)
3. [PSSM Profiles — Position-Specific Scoring Matrices](#3-pssm-profiles)
4. [Hyperparameter Tuning — Finding the Best Model Systematically](#4-hyperparameter-tuning)
5. [Putting It Together — PSSM + Your Best Model](#5-putting-it-together)

---

## 1. Why Evolution Helps

### The core insight

Two proteins from different species that share a common ancestor will diverge over millions of years. Mutations accumulate randomly, but **only mutations that do not break the protein's function survive**. This means that positions which are *critical* for structure or function — the ones that, if mutated, would cause the protein to misfold or lose its activity — remain conserved across many species. Positions that are *not* critical vary freely.

The implication for prediction is profound: **if you can see which positions are conserved across hundreds of homologs, you already know a great deal about which positions are structurally constrained** — and structurally constrained positions are much easier to label as helix, strand, or coil.

A single protein sequence tells you what the amino acids *are* at each position today. An alignment of many homologs tells you what they *could have been* and chose not to be — which is often more informative.

### A concrete example

Consider position 57 of a trypsin from various mammals:

```
Human:    ...L  S  [A]  G  H...
Bovine:   ...L  S  [A]  G  Y...
Pig:      ...L  S  [A]  G  H...
Mouse:    ...L  S  [A]  G  H...
Rat:      ...L  S  [A]  G  H...
Zebrafish:...L  S  [A]  G  N...
```

Position 57 is Alanine in every species. That near-perfect conservation is a strong signal: this residue is under selection, and a model that knows "this position is always A" carries extra information beyond the raw sequence. Meanwhile the adjacent position (4th column) is variable — H, Y, N — and carries less structural signal.

This logic is the foundation of everything in this week.

---

## 2. Multiple Sequence Alignment as a Profile

### From a pair to a crowd

In Week 1 you aligned *two* sequences to find conserved residues. Now you align **hundreds** of homologs at once — a **Multiple Sequence Alignment (MSA)** — and extract, for every column (= every position in your query protein), a histogram of which amino acids appear there.

```
Query:       M  N  I  F  E  M  L  R  I  D
Homolog 1:   M  N  I  F  E  M  L  R  I  D
Homolog 2:   M  -  I  F  E  M  L  R  V  D
Homolog 3:   M  N  I  Y  E  M  I  R  I  D
Homolog 4:   M  N  V  F  E  M  L  K  I  D
             ↓
At position 4 (F): F=75%, Y=25% — nearly conserved
At position 9 (I): I=75%, V=25% — conservative variation (both hydrophobic)
At position 8 (R): R=75%, K=25% — conservative (both positively charged)
```

This per-column histogram, normalised and log-scaled, is the **profile**. A profile converts a single-sequence row into a **matrix of shape (L, 20)** — one row per residue, one column per amino acid — that encodes evolutionary frequency at each position.

### Tools for building an MSA from homologs

You do not build the MSA by hand. Standard tools search a large sequence database for homologs and return an aligned set:

| Tool | What it does | Speed / sensitivity |
|------|-------------|---------------------|
| **PSI-BLAST** | Iterative BLAST: finds homologs, builds a profile, re-searches with that profile | Standard; slower but familiar |
| **HHblits** | HMM-vs-HMM search; finds very distant homologs | Faster, more sensitive for remote homologs |
| **jackhmmer** | Iterative HMMER search; similar to PSI-BLAST in spirit | Widely used in research pipelines |

For this course, **PSI-BLAST** is the most accessible because it runs on the NCBI web server with no installation, and Biopython can wrap it programmatically.

→ [PSI-BLAST web server (NCBI)](https://blast.ncbi.nlm.nih.gov/Blast.cgi?PROGRAM=blastp&PAGE_TYPE=BlastSearch&BLAST_SPEC=blast2seq)  
→ [HHblits (MPI Bioinformatics Toolkit)](https://toolkit.tuebingen.mpg.de/tools/hhblits)  
→ [HMMER / jackhmmer](https://www.ebi.ac.uk/Tools/hmmer/search/jackhmmer)

---

## 3. PSSM Profiles

### What is a PSSM?

A **Position-Specific Scoring Matrix (PSSM)** is the standard way to turn an MSA into a per-residue numeric feature. At position `i`, instead of a single amino acid, you have a 20-number vector of **log-odds scores** — one per amino acid — that says how much more or less likely each amino acid is at this position compared to its background frequency in all proteins.

Formally, for position `i` and amino acid `a`:

```
PSSM[i][a] = log2( f(i,a) / b(a) )
```

where:
- `f(i, a)` = frequency of amino acid `a` in column `i` of the MSA (smoothed to avoid log-zero)
- `b(a)` = background frequency of amino acid `a` in all proteins (from BLOSUM)

| PSSM value | Meaning |
|------------|---------|
| **Large positive** (e.g. +4) | Amino acid `a` appears far more often here than expected by chance — strong conservation |
| **Near zero** | About as common as expected — no strong preference |
| **Large negative** (e.g. −4) | Amino acid `a` is avoided at this position — likely physically or functionally incompatible |

The 20 PSSM values at a position tell you the evolutionary "personality" of that site in a way one-hot encoding cannot — they encode *which alternatives are tolerated*.

### Getting a PSSM with PSI-BLAST

PSI-BLAST runs several iterations. In each iteration it searches a database, collects significant hits, builds a profile, and uses that profile for the next search — finding increasingly distant homologs with each round. The final output includes the PSSM.

**From the web** — go to BLAST, run PSI-BLAST for 3 iterations, download the "PSSM" from the results page.

**Programmatically with Biopython:**

```python
from Bio.Blast.Applications import NcbipsiblastCommandline
import numpy as np

# Run PSI-BLAST (requires local BLAST+ installation)
psiblast = NcbipsiblastCommandline(
    query="my_protein.fasta",
    db="swissprot",          # or nr — larger db gives more homologs
    num_iterations=3,
    out_ascii_pssm="my_protein.pssm",
    outfmt=5,
    out="blast_output.xml"
)
stdout, stderr = psiblast()
```

**Parsing the `.pssm` file:**

The PSSM file is a fixed-width text file. Each data row corresponds to one residue; columns 3–22 are the 20 log-odds scores in the order `A C D E F G H I K L M N P Q R S T V W Y`.

```python
def parse_pssm(pssm_path):
    """Return a (L, 20) numpy array of PSSM scores."""
    rows = []
    with open(pssm_path) as f:
        for line in f:
            parts = line.split()
            # data lines start with an integer (residue number)
            if len(parts) >= 22 and parts[0].isdigit():
                scores = list(map(int, parts[2:22]))
                rows.append(scores)
    return np.array(rows, dtype=np.float32)   # shape (L, 20)

pssm = parse_pssm("my_protein.pssm")
print(pssm.shape)   # (L, 20)
```

### Combining PSSM with existing features

PSSM values slot in naturally as an **additional 20 features per residue**, on top of the features you already have:

| Feature set | Dimension per residue | What it encodes |
|-------------|----------------------|-----------------|
| One-hot | 20 | Identity only |
| One-hot + PSSM | 40 | Identity + evolutionary conservation |
| Sliding window one-hot | 17 × 20 = 340 | Local context |
| Sliding window (one-hot + PSSM) | 17 × 40 = 680 | Local context + evolution at every window position |
| ESM2 embedding | 320 | Learned context (includes implicit evolutionary signal) |

For your classical models (Week 3), concatenate the PSSM row to the one-hot vector for each residue before building the sliding-window matrix.

For your deep models (Week 4), concatenate the PSSM tensor to the embedding at each position:

```python
# pssm: (L, 20) tensor; emb: (B, L, d) tensor from nn.Embedding
import torch

pssm_tensor = torch.tensor(pssm).unsqueeze(0)          # (1, L, 20)
pssm_normed = (pssm_tensor - pssm_tensor.mean()) / (pssm_tensor.std() + 1e-8)
combined = torch.cat([emb, pssm_normed.expand(B, -1, -1)], dim=-1)  # (B, L, d+20)
```

Normalise the PSSM values (zero mean, unit variance) before concatenating — raw PSSM scores span roughly −6 to +11 and will dominate the embedding if left unscaled.

### How much does PSSM help?

On the standard CB513 benchmark for Q3 prediction:

| Input features | Typical Q3 accuracy |
|---------------|---------------------|
| One-hot sequence only (classical) | ~65–70% |
| One-hot sequence only (deep BiLSTM) | ~72–76% |
| Sequence + PSSM (classical) | ~72–76% |
| Sequence + PSSM (deep BiLSTM) | ~78–82% |
| ESM2 / ProtT5 embeddings (deep) | ~85–90% |

Adding PSSM to a classical model often matches or beats a deep model on raw sequence alone — this is why evolutionary features dominated the field for two decades before protein language models arrived.

→ [Original PSSM / PSI-BLAST paper — Altschul et al. (1997)](https://www.sciencedirect.com/science/article/pii/S0022283697914870)  
→ [Biopython BLAST Applications docs](https://biopython.org/docs/latest/Tutorial/chapter_blast.html)  
→ [PSIPRED — one of the most influential PSSM-based secondary structure predictors](https://bioinf.cs.ucl.ac.uk/psipred/)

---

## 4. Hyperparameter Tuning

### Parameters vs hyperparameters

Every model has two kinds of settings:

| Kind | What they are | How they are learned |
|------|--------------|---------------------|
| **Parameters** | Weights, biases — the numbers the model adjusts during training | Gradient descent / MLE / counting |
| **Hyperparameters** | Decisions made *before* training: architecture, regularisation strength, learning rate, window size, etc. | **You** set them, or search over them |

Picking hyperparameters by hand ("I'll try C=1 and see") is guessing. Systematic search gives you the best model you can build, and is the difference between a research-quality result and a first attempt.

### The two rules that must not break

Both rules carry over from Weeks 3 and 4:

1. **Never tune on the test set.** The test set is seen once, at the very end, to report your final number. If you use it to pick hyperparameters, your "test" score is actually a training score — you have overfit your choices to the test data without knowing it.

2. **Split by protein chain, not by residue.** The same data-leakage risk from Weeks 3 and 4 applies to the validation fold used during tuning.

The correct setup: **train / validation / test split by chain**, where:
- **train** — used to fit the model's parameters
- **validation** — used to evaluate each hyperparameter configuration
- **test** — touched once, at the very end

```python
from sklearn.model_selection import GroupShuffleSplit

gss = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=0)
dev_idx, test_idx = next(gss.split(X, y, groups=chain_ids))

gss2 = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=0)
train_idx, val_idx = next(gss2.split(X[dev_idx], y[dev_idx], groups=chain_ids[dev_idx]))
```

### Grid search

Try every combination of a specified set of values. Exhaustive but exponential in the number of hyperparameters — fine when you have a small search space.

```python
from sklearn.model_selection import GridSearchCV, GroupKFold

param_grid = {
    "C":      [0.01, 0.1, 1.0, 10.0],
    "max_iter": [1000],
}

from sklearn.svm import LinearSVC
clf = GridSearchCV(
    LinearSVC(),
    param_grid,
    cv=GroupKFold(n_splits=5),    # group-aware cross-validation!
    scoring="f1_macro",
    n_jobs=-1,
)
clf.fit(X_train, y_train, groups=chain_ids_train)
print(clf.best_params_, clf.best_score_)
```

### Random search

Sample hyperparameter combinations at random from a defined distribution. Often finds good configurations faster than grid search when the search space is large, because many hyperparameters have little effect — random search spends the same budget exploring the ones that matter.

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import loguniform, randint

param_dist = {
    "n_estimators": randint(100, 600),
    "max_depth":    [None, 5, 10, 20],
    "max_features": ["sqrt", "log2"],
    "min_samples_leaf": randint(1, 20),
}

from sklearn.ensemble import RandomForestClassifier
search = RandomizedSearchCV(
    RandomForestClassifier(class_weight="balanced", n_jobs=-1),
    param_dist,
    n_iter=40,
    cv=GroupKFold(n_splits=5),
    scoring="f1_macro",
    random_state=42,
)
search.fit(X_train, y_train, groups=chain_ids_train)
print(search.best_params_)
```

> **Rule of thumb.** For ≤ 3 hyperparameters with ≤ 4 values each: grid search. For ≥ 4 hyperparameters or continuous ranges: random search with 30–50 iterations.

### Bayesian optimisation with Optuna (for deep models)

For neural networks, grid and random search are wasteful — training one configuration takes minutes. **Bayesian optimisation** builds a surrogate model of the loss landscape and proposes the *most promising* configuration to try next, reducing the number of trials needed to find a good optimum.

[**Optuna**](https://optuna.org/) is the simplest library for this:

```python
import optuna, torch, torch.nn as nn

def objective(trial):
    lr      = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    dropout = trial.suggest_float("dropout", 0.0, 0.5)
    hidden  = trial.suggest_categorical("hidden", [32, 64, 128])
    layers  = trial.suggest_int("layers", 1, 3)

    model = BiLSTM(hidden=hidden, layers=layers, dropout=dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    val_q3 = train_and_eval(model, optimizer, train_loader, val_loader)
    return val_q3   # Optuna maximises this

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=30)
print(study.best_params)
```

→ [Optuna documentation](https://optuna.readthedocs.io/)  
→ [Bergstra & Bengio (2012) — Why random search beats grid search](https://jmlr.org/papers/v13/bergstra12a.html)

### Learning curves — diagnosing over- and underfitting

Before tuning is complete, plot the **learning curve**: training F1 and validation F1 as a function of training set size (or training epochs for deep models). The shape tells you exactly what to do next:

```
Validation F1
     │
high ├───────────────── train F1
     │             ╱
     │        ────╱
     │   ────╱
low  │────╱____________________
     └──────────────────────── Training set size

  Both curves are high and converging:
  → good generalisation, you are done (or can try larger model)

  Large gap between train and val:
  → overfitting → increase regularisation (C, dropout, max_depth, weight_decay)
                → more training data

  Both curves are low and converging:
  → underfitting → more expressive model, more features, longer training
```

For deep models, plot train loss and validation Q3 accuracy per epoch instead of per dataset size — the same diagnosis applies.

```python
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve

sizes, tr_scores, val_scores = learning_curve(
    best_clf, X_train, y_train,
    groups=chain_ids_train,
    cv=GroupKFold(n_splits=5),
    train_sizes=np.linspace(0.1, 1.0, 8),
    scoring="f1_macro",
    n_jobs=-1,
)
plt.plot(sizes, tr_scores.mean(1),  label="Train F1")
plt.plot(sizes, val_scores.mean(1), label="Val F1")
plt.xlabel("Training chains"); plt.ylabel("Macro-F1"); plt.legend()
```

### Key hyperparameters per model family

| Model | Most important hyperparameters | Start here |
|-------|-------------------------------|-----------|
| **LinearSVC** | `C` (regularisation inverse) | log-scale: 0.01, 0.1, 1, 10 |
| **Random Forest** | `n_estimators`, `max_depth`, `min_samples_leaf` | 200 trees, no depth limit, then tighten |
| **Naive Bayes** | `alpha` (Laplace smoothing) | 0.01, 0.1, 1.0 |
| **BiLSTM** | `lr`, `hidden`, `layers`, `dropout` | lr=1e-3, hidden=64, layers=2, dropout=0.2 |
| **Transformer** | `lr`, `d_model`, `nhead`, `layers`, `dropout` | lr=1e-3, d=64, 4 heads, 2 layers |
| **CNN** | `lr`, channels, `kernel_size` | lr=1e-3, 64 channels, kernel 7 |

---

## 5. Putting It Together

### The complete Week 5 pipeline

```
protein sequences + DSSP labels (from Week 2)
         |
         |  PSI-BLAST (3 iterations) on each sequence
         v
   PSSM files: one per protein, shape (L, 20)
         |
         |  normalise PSSM values (zero mean, unit std)
         v
   combined per-residue feature:
      classical  ->  sliding window (one-hot + PSSM) shape (L, 17*40)
      deep       ->  concat PSSM to embedding at each position shape (B, L, d+20)
         |
         |  group-aware train / val / test split by chain
         v
   systematic tuning:
      classical  ->  GridSearchCV or RandomizedSearchCV with GroupKFold
      deep       ->  Optuna with 20–30 trials
         |
         v
   retrain best config on full train+val set
         |
         v
   evaluate on held-out test chains  -->  Q3 accuracy (macro-F1 for classical)
                                          compare:  Wk3  |  Wk4  |  Wk5 (PSSM)
```

### What to expect

Adding a PSSM typically gives:
- **+5–8 percentage points** in macro-F1 for classical models over one-hot alone.
- **+2–4 percentage points** in Q3 accuracy for deep models (the embedding already encodes some evolutionary information implicitly, so the gain is smaller).

Hyperparameter tuning on top of that typically adds another **+2–3 points** — smaller, but meaningful when models are close.

> **Looking ahead to Week 6.** Next week you will combine the best-performing model and feature set from Weeks 1–5 into a complete, reproducible end-to-end pipeline: fetch a sequence, compute PSSM, encode features, run inference, and interpret the per-residue predictions biologically. Everything you have built — the DSSP labels, the classical and deep models, the PSSM features, the tuned hyperparameters — feeds directly into that final deliverable.

---

## Resources

- [PSI-BLAST — Altschul et al. (1997)](https://www.sciencedirect.com/science/article/pii/S0022283697914870) — the original paper introducing iterative profile search
- [PSIPRED server](https://bioinf.cs.ucl.ac.uk/psipred/) — one of the most accurate classical secondary structure predictors; powered by PSSM features
- [NCBI PSI-BLAST web server](https://blast.ncbi.nlm.nih.gov/Blast.cgi?PROGRAM=blastp&PAGE_TYPE=BlastSearch) — run PSI-BLAST online and download a PSSM file
- [Biopython BLAST tutorial](https://biopython.org/docs/latest/Tutorial/chapter_blast.html) — parsing BLAST output and running searches from Python
- [Optuna documentation](https://optuna.readthedocs.io/) — Bayesian hyperparameter optimisation
- [scikit-learn: Tuning estimator hyperparameters](https://scikit-learn.org/stable/modules/grid_search.html) — GridSearchCV, RandomizedSearchCV, and learning curves
- [Bergstra & Bengio (2012)](https://jmlr.org/papers/v13/bergstra12a.html) — the paper that established random search outperforms grid search
- [HHblits (MPI Bioinformatics Toolkit)](https://toolkit.tuebingen.mpg.de/tools/hhblits) — faster, more sensitive alternative to PSI-BLAST for finding distant homologs
- [Protein Secondary Structure dataset (Kaggle)](https://www.kaggle.com/datasets/alfrandom/protein-secondary-structure) — the same labelled dataset from Weeks 3 and 4

---

[← Home](../../README.md)
