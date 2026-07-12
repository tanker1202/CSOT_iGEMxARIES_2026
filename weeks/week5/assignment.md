# Week 5 Assignment — PSSM Features & Hyperparameter Tuning

[← Week 5 Topics](index.md) | [Leaderboard →](../../leaderboard.md)

---

## Task

Augment your best model from Weeks 3 or 4 with **evolutionary PSSM features**, then apply **systematic hyperparameter tuning** to push its Q3 accuracy as high as you can. Submit a report comparing your new results against your previous baselines.

> This week's assignment bridges everything you have built so far. The PSSM features you add here are the same kind of evolutionary signal that tools like PSIPRED used to lead the field for two decades. Seeing how much they help — and how tuning stacks on top — sets the stage for the final pipeline in Week 6.

---

## The Dataset

Continue using the same [RCSB Kabsch–Sander dataset](https://www.kaggle.com/datasets/alfrandom/protein-secondary-structure) from Weeks 3 and 4. This week you work with a **small subset of 20–50 chains** for the PSSM step, because running PSI-BLAST on thousands of proteins is not feasible within a week.

**How to pick your subset:**

- Pull 20–50 chains at random from your existing train split (use `random.seed(42)` for reproducibility).
- Keep only chains between 50–300 residues so the web BLAST runs quickly.
- Record the `pdb_id` and `chain_code` so you can re-identify them later.

**Alternative** — if you would prefer pre-computed PSSMs without running BLAST, the [CB513 benchmark set](https://github.com/alrojo/cb513) provides PSSMs alongside sequences and labels. You may use it instead; if you do, note which set you used and ensure your train/held-out split is still chain-level.

---

## Deliverables

### 1. PSSM Extraction & Inspection

Run PSI-BLAST (3 iterations, against Swiss-Prot or nr) on each protein in your 20–50 chain subset and save the resulting `.pssm` file. You may use:

- The **NCBI BLAST web server** → run PSI-BLAST → click "Download PSSM" after the final iteration.
- Or the **Biopython programmatic wrapper** (requires a local BLAST+ installation — see code snippet below).

Parse each `.pssm` file into a `(L, 20)` NumPy array, then report:

- The shape of one example PSSM matrix.
- The **3 most conserved positions** in that protein — defined as the positions with the highest sum of absolute PSSM values across all 20 amino acids (a large total magnitude means the model has strong preferences). For each: the residue, the position number, its Q3 label, the most preferred amino acid (highest positive score), and the most avoided amino acid (largest negative score).
- **1–2 sentences** connecting what you see to the biology: do the conserved positions correspond to helix or strand residues? Are the preferred amino acids consistent with known secondary-structure propensities?

### 2. PSSM Feature Integration

Add the PSSM as an additional 20 features per residue and build the combined feature representation for your PSSM subset. You must implement **one of the two** integration paths below — pick the one that matches your best model from last week:

**Path A — classical model (from Week 3):**  
Concatenate the PSSM row to the one-hot vector at each position, then build the sliding-window matrix. For a window of size 17 and combined per-residue dimension of 40 (20 one-hot + 20 PSSM), the resulting matrix has shape `(L, 17 × 40) = (L, 680)`. Report this shape.

**Path B — deep model (from Week 4):**  
After the embedding layer, concatenate the (normalised) PSSM tensor to the per-residue representation before the main model body. Report the new per-residue input dimensionality (`d_embedding + 20`).

Then report, for your PSSM subset, the **mean PSSM value per Q3 class** (H, E, C) across all 20 amino acid channels — i.e., average each of the 20 PSSM columns over all residues in each class. Present this as a small table (3 rows × 20 columns is fine). In **2–3 sentences**, describe any pattern: do helix residues tend to have different mean PSSM profiles than strand residues?

### 3. Systematic Hyperparameter Tuning

Apply at least one systematic tuning method to your best model. Run tuning **twice** — once on sequence-only features (your Week 3/4 baseline features) and once with PSSM-augmented features — so you can isolate the contribution of each.

**For classical models:** use `GridSearchCV` or `RandomizedSearchCV` with `GroupKFold(n_splits=5)` keyed on chain ID. Tune at least two hyperparameters — refer to the per-model cheat sheet in the Week 5 notes.

**For deep models:** use Optuna with at least 20 trials. The objective function must return held-out Q3 accuracy on a chain-level validation split (not a per-residue random split). Tune at least: learning rate, hidden dimension, and dropout.

For each tuning run report:

- The hyperparameter search space used.
- The best hyperparameter configuration found.
- A **plot or table** of validation score vs trial / parameter value that shows the search converging (a plot of `val_score` vs trial index for Optuna, or the `cv_results_` from sklearn).
- A **learning curve** (train and validation Q3/F1 vs training-set size or epoch) for the final best configuration.

### 4. Results Comparison

Collect all your numbers in one place:

- A results table with at least four rows: **Week 3 classical baseline**, **Week 4 deep baseline**, **Best model + PSSM (untuned)**, **Best model + PSSM (tuned)**. Columns: model name, feature set, held-out Q3 accuracy (or macro-F1 if classical), held-out strand (E) accuracy.
- A **confusion matrix over H / E / C** for your best overall result.
- **3–5 sentences** interpreting the table: how much did PSSM add over sequence alone? How much did tuning add on top of PSSM? Which class benefited most? Is the improvement consistent with the expected gains listed in the Week 5 notes?

### 5. Methodology Statement

In **2–3 sentences**, confirm:

- Your chain-level split is preserved throughout — PSSM extraction was done only for proteins in the train split; no held-out chain contributed to fitting the vectoriser, the PSSM parser, or any hyperparameter decision.
- Tuning used `GroupKFold` (classical) or a chain-level validation split (deep) — held-out chains were never touched during the search.

---

## Recommended Workflow

```
1. Sample 20–50 chains from your existing train split (random.seed(42))

2. Run PSI-BLAST for each chain (web server or Biopython CLI wrapper)
   -> save one .pssm file per chain

3. Parse .pssm files -> dict mapping chain_id to (L, 20) numpy array

4. Inspect PSSMs (Deliverable 1)

5. Integrate PSSM into your feature pipeline
   Path A (classical): concatenate to one-hot per position -> sliding window
   Path B (deep): normalise + concatenate to embedding output

6. Tune hyperparameters
   Classical: GridSearchCV / RandomizedSearchCV with GroupKFold
   Deep: Optuna, 20+ trials, chain-level val split inside objective

7. Collect results: Wk3 baseline | Wk4 baseline | +PSSM untuned | +PSSM tuned

8. Write up (deliverables 1-5)
```

### Code snippets

**Installation:**

```
pip install biopython numpy pandas scikit-learn optuna
pip install torch   # if using deep path
```

**Running PSI-BLAST programmatically (requires BLAST+ installed locally):**

```python
from Bio.Blast.Applications import NcbipsiblastCommandline

def run_psiblast(fasta_path, pssm_out, db="swissprot"):
    cmd = NcbipsiblastCommandline(
        query=fasta_path,
        db=db,
        num_iterations=3,
        out_ascii_pssm=pssm_out,
        outfmt=5,
        out="/dev/null",    # discard the XML; we only want the PSSM
        num_threads=4,
    )
    cmd()

# If using the web server instead, download the PSSM file manually after
# the third PSI-BLAST iteration and save it as <chain_id>.pssm
```

**Parsing a `.pssm` file:**

```python
import numpy as np

PSSM_AA_ORDER = list("ACDEFGHIKLMNPQRSTVWY")

def parse_pssm(path):
    rows = []
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 22 and parts[0].isdigit():
                rows.append(list(map(int, parts[2:22])))
    return np.array(rows, dtype=np.float32)   # (L, 20)
```

**Path A — PSSM + one-hot sliding window (classical):**

```python
import numpy as np

AA_ORDER = "ACDEFGHIKLMNPQRSTVWY"
aa_to_idx = {a: i for i, a in enumerate(AA_ORDER)}

def one_hot(aa):
    v = np.zeros(20)
    if aa in aa_to_idx: v[aa_to_idx[aa]] = 1.0
    return v

def sliding_window_pssm(sequence, pssm, window=17):
    """pssm: (L, 20) numpy array, already normalised."""
    pad = window // 2
    padded_seq  = "-" * pad + sequence + "-" * pad
    padded_pssm = np.zeros((pad + len(sequence) + pad, 20))
    padded_pssm[pad: pad + len(sequence)] = pssm
    features = []
    for i in range(len(sequence)):
        oh  = np.concatenate([one_hot(padded_seq[i + j]) for j in range(window)])
        ps  = padded_pssm[i: i + window].flatten()
        features.append(np.concatenate([oh, ps]))
    return np.array(features)   # (L, window * 40)
```

**Path B — PSSM concatenated to embedding (deep):**

```python
import torch

def add_pssm_to_batch(emb, pssm_batch):
    """
    emb:       (B, L, d)   — output of nn.Embedding
    pssm_batch:(B, L, 20)  — pre-normalised PSSM tensors, padded to match L
    Returns:   (B, L, d+20)
    """
    pssm_normed = (pssm_batch - pssm_batch.mean(dim=1, keepdim=True)) / \
                  (pssm_batch.std(dim=1, keepdim=True) + 1e-8)
    return torch.cat([emb, pssm_normed], dim=-1)
```

**Hyperparameter tuning — classical (GroupKFold):**

```python
from sklearn.model_selection import RandomizedSearchCV, GroupKFold
from sklearn.svm import LinearSVC
from scipy.stats import loguniform

search = RandomizedSearchCV(
    LinearSVC(class_weight="balanced", max_iter=2000),
    param_distributions={"C": loguniform(1e-3, 1e2)},
    n_iter=30,
    scoring="f1_macro",
    cv=GroupKFold(n_splits=5),
    random_state=42,
    n_jobs=-1,
)
search.fit(X_train, y_train, groups=chain_ids_train)
print(search.best_params_, search.best_score_)
```

**Hyperparameter tuning — deep (Optuna):**

```python
import optuna, torch

def objective(trial):
    lr      = trial.suggest_float("lr",      1e-4, 5e-3, log=True)
    hidden  = trial.suggest_categorical("hidden", [32, 64, 128])
    dropout = trial.suggest_float("dropout", 0.0, 0.4)
    layers  = trial.suggest_int("layers", 1, 3)

    model = BiLSTM(hidden=hidden, layers=layers, dropout=dropout)
    opt   = torch.optim.Adam(model.parameters(), lr=lr)
    return train_and_eval(model, opt, train_loader, val_loader)   # returns val Q3

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=25)
print(study.best_params)
```

**Learning curve (classical):**

```python
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve

sizes, tr, va = learning_curve(
    search.best_estimator_, X_train, y_train,
    groups=chain_ids_train,
    cv=GroupKFold(n_splits=5),
    train_sizes=[0.1, 0.2, 0.4, 0.6, 0.8, 1.0],
    scoring="f1_macro", n_jobs=-1,
)
plt.plot(sizes, tr.mean(1), label="Train"); plt.plot(sizes, va.mean(1), label="Val")
plt.xlabel("Training chains"); plt.ylabel("Macro-F1"); plt.legend()
```

---

## Optional (ungraded)

*No points — for depth and to preview Week 6.*

- **HHblits instead of PSI-BLAST.** HHblits finds more distant homologs and often gives richer PSSMs. Run it on a few proteins and compare the PSSM magnitudes to those from PSI-BLAST.
- **Information content.** At each column of the MSA, compute the Shannon entropy `H = -Σ f·log(f)`. Plot information content along the sequence; do the low-entropy (highly conserved) positions correspond to helix/strand or coil?
- **PSSM-only model.** Train a model using *only* the PSSM features (no one-hot). Compare to one-hot-only and one-hot+PSSM. What does this tell you about how much evolutionary information alone can predict structure?

---

## Submission Format

Submit a single PDF or Markdown report (with code as a `.py` file or notebook attached) containing:

- [ ] List of chains used for PSSM extraction (20–50, with chain IDs); which database and how many PSI-BLAST iterations
- [ ] Shape of one PSSM matrix; table of 3 most conserved positions with preferred/avoided amino acids; 1–2 sentence biological interpretation
- [ ] Combined feature dimensionality (Path A or B) with mean PSSM per Q3 class; 2–3 sentence pattern description
- [ ] Hyperparameter search space, best configuration, convergence plot, and learning curve — for both sequence-only and PSSM-augmented runs
- [ ] Results table (≥4 rows: Wk3 baseline, Wk4 baseline, +PSSM untuned, +PSSM tuned) + confusion matrix + 3–5 sentence interpretation
- [ ] Methodology statement (chain-level split preserved; no held-out leakage during tuning)
- [ ] Code (`.py` file or notebook)

**Deadline**: Jul 14, 2026 — results posted to the [leaderboard](../../leaderboard.md) within 48 hours.

---

## Marking Criteria

| Component | Weight | What we look for |
|-----------|--------|-----------------|
| **Correctness** | 60% | PSSM parsed and interpreted correctly; features integrated without shape errors or leakage; tuning uses a group-aware CV; results table is internally consistent and improvements are real |
| **Efficiency** | 20% | PSSM parsing is clean and reusable; features are not redundantly recomputed; tuning budget (trials / grid size) is proportionate; code is readable |
| **Clarity and Explanation** | 20% | Well-structured report; PSSM inspection connects to biology; convergence plots are readable; results table is clearly formatted with a sound interpretation of what each ingredient contributed |

### Correctness breakdown (60 pts)

| Sub-component | Points |
|---------------|--------|
| PSSM extraction correct and conserved-position analysis sound | 15 |
| PSSM integrated into features without dimension errors or train/held-out leakage | 15 |
| Tuning is group-aware and convergence evidence is provided | 15 |
| Results table complete and comparison to Weeks 3/4 baselines is valid | 15 |

---

[← Week 5 Topics](index.md) | [Leaderboard →](../../leaderboard.md)
