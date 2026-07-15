# Week 6 — End-to-End Pipeline, Error Analysis & Biological Inference

[← Home](../../README.md)

---

> **The final week.** You have built every component: sequence databases (Week 1), structure labels and feature encodings (Week 2), classical models (Week 3), deep networks (Week 4), and evolutionary PSSM features with systematic tuning (Week 5). This week you wire them into a single, reproducible pipeline; learn to read *where* and *why* your model fails; and then step back and ask the question the whole course has been building toward — *what does a secondary-structure prediction actually tell you about the biology of a protein?*

---

## Topics

1. [The End-to-End Pipeline — From Sequence to Prediction](#1-the-end-to-end-pipeline)
2. [Error Analysis & Iteration — Understanding and Fixing Failures](#2-error-analysis--iteration)
3. [Inferring Biological Properties from Model Outputs](#3-inferring-biological-properties)

---

## 1. The End-to-End Pipeline

### What "end-to-end" means

The previous weeks each handed you a component: a feature encoder, a trained model, an evaluation loop. An **end-to-end pipeline** is a single, callable piece of code that accepts a protein sequence (or a FASTA file) and returns a per-residue secondary-structure prediction string — with no manual steps in between. This is what separates a research experiment from a usable tool.

A good pipeline has four properties:

| Property | What it means |
|----------|--------------|
| **Reproducible** | Given the same input and the same saved model, it always produces the same output |
| **Modular** | Each step is a function; swapping the model or the feature encoder does not touch the other steps |
| **Self-contained** | All preprocessing parameters (the vectoriser, the PSSM scaler) were fitted on training data and are saved alongside the model weights |
| **Transparent** | The pipeline logs which model, which features, and which hyperparameters were used to produce each prediction |

### The five stages

```
┌─────────────────────────────────────────────────────────────────────┐
│  Stage 1  │  Sequence acquisition                                   │
│           │  Read a FASTA file, or fetch from UniProt / NCBI        │
├─────────────────────────────────────────────────────────────────────┤
│  Stage 2  │  PSSM computation   (optional but recommended)          │
│           │  Run PSI-BLAST → parse .pssm → normalise                │
├─────────────────────────────────────────────────────────────────────┤
│  Stage 3  │  Feature encoding                                       │
│           │  one-hot sliding window + PSSM  OR  ESM2 embedding      │
├─────────────────────────────────────────────────────────────────────┤
│  Stage 4  │  Model inference                                        │
│           │  Load saved model → forward pass → per-residue logits   │
├─────────────────────────────────────────────────────────────────────┤
│  Stage 5  │  Post-processing & output                               │
│           │  Argmax logits → Q3 string → optional smoothing         │
└─────────────────────────────────────────────────────────────────────┘
```

### Saving and loading your trained model

Nothing is truly end-to-end if you have to retrain before each prediction. Save every artefact that was fitted on training data — the model weights, the feature scaler, and the text vectoriser — immediately after training, and load them at inference time.

**Saving (after training):**

```python
import pickle, torch

# Classical model (sklearn)
with open("model.pkl", "wb") as f:
    pickle.dump(best_clf, f)

with open("tfidf_vectoriser.pkl", "wb") as f:
    pickle.dump(fitted_tfidf, f)

# Deep model (PyTorch)
torch.save(model.state_dict(), "bilstm.pt")
torch.save({"hidden": 64, "layers": 2, "dropout": 0.2}, "bilstm_config.json")
```

**Loading and running inference:**

```python
import pickle, torch

# Classical
with open("model.pkl", "rb") as f:
    clf = pickle.load(f)
with open("tfidf_vectoriser.pkl", "rb") as f:
    vec = pickle.load(f)

# Deep
import json
cfg = json.load(open("bilstm_config.json"))
model = BiLSTM(**cfg)
model.load_state_dict(torch.load("bilstm.pt", map_location="cpu"))
model.eval()
```

### A complete pipeline class

The code below is a blueprint. Adapt it to whichever model (classical or deep) and features (one-hot, PSSM, ESM2) you settled on in Week 5.

```python
import numpy as np

Q3_LABELS = {0: "H", 1: "E", 2: "C"}

class SecStructPredictor:
    """Wraps a trained model and all preprocessing into a single callable."""

    def __init__(self, model, feature_fn, label_map=Q3_LABELS):
        """
        model      : fitted sklearn estimator OR eval-mode PyTorch module
        feature_fn : callable  sequence [+ pssm]  ->  feature array (L, d)
        """
        self.model      = model
        self.feature_fn = feature_fn
        self.label_map  = label_map

    def predict(self, sequence, pssm=None):
        X = self.feature_fn(sequence, pssm)             # (L, d)
        if hasattr(self.model, "predict"):               # sklearn path
            indices = self.model.predict(X)
        else:                                            # PyTorch path
            import torch
            with torch.no_grad():
                t = torch.tensor(X).unsqueeze(0).float()
                logits = self.model(t)                   # (1, L, C)
                indices = logits.argmax(-1)[0].numpy()
        return "".join(self.label_map[i] for i in indices)

    def predict_from_fasta(self, fasta_path, pssm_dir=None):
        """
        Returns a list of (header, sequence, prediction_string) tuples.
        """
        from Bio import SeqIO
        results = []
        for record in SeqIO.parse(fasta_path, "fasta"):
            seq = str(record.seq)
            pssm = None
            if pssm_dir:
                import os
                pssm_path = os.path.join(pssm_dir, f"{record.id}.pssm")
                if os.path.exists(pssm_path):
                    pssm = parse_pssm(pssm_path)        # from Week 5
            pred = self.predict(seq, pssm)
            results.append((record.description, seq, pred))
        return results

    def save_predictions(self, results, out_path):
        """Save (header, seq, pred) triples to a TSV file."""
        with open(out_path, "w") as f:
            f.write("header\tsequence\tprediction\n")
            for header, seq, pred in results:
                f.write(f"{header}\t{seq}\t{pred}\n")
```

### Output format

Predictions can be represented in several ways. The simplest is a **secondary-structure string** that mirrors the sequence:

```
Sequence:   MNIFEMLRIDEKLGLGTSFPVHLT
Prediction: CCCHHHHHHHHCCCEEECCCCCCC
```

One column per residue, trivial to parse, trivial to align against the sequence. For downstream analysis, save it alongside the sequence as a TSV or in FASTA-like format (use `>pred|<id>` as the header on the prediction line).

### Reproducibility checklist

Before calling your pipeline "done", verify:

```
[ ] Model weights saved with the hyperparameter config that produced them
[ ] Vectoriser / scaler fitted only on training chains, saved and loaded at test time
[ ] Random seed set (numpy, torch, python random) before any split or training
[ ] The split itself is saved (or deterministically reproducible from the seed)
[ ] Running the pipeline twice on the same input produces the same output
[ ] The prediction file is stamped with which model file and dataset were used
```

---

## 2. Error Analysis & Iteration

### Why accuracy alone is not enough

After five weeks of training and evaluation, you have a Q3 accuracy number. But a number without analysis is nearly useless for improvement. The goal of error analysis is to answer one question: **which residues does the model get wrong, and why?**

Understanding that drives every useful improvement from here.

### Reading the confusion matrix properly

The Q3 confusion matrix is a 3×3 table of truth vs prediction. Do not just look at the diagonal — the off-diagonal cells are where the story is.

```
              Predicted
              H     E     C
Actual  H  [ 812   23   165 ]
        E  [  45  301   189 ]
        C  [  89   67  2340 ]
```

Read each row as: "of all true X residues, how many did the model label H / E / C?" The above shows that:
- True H residues: the model is fairly good at helices (812/1000 = 81%), but 16.5% leak into C.
- True E residues: strands are harder (301/535 = 56%), with errors splitting roughly equally between H and C.
- True C residues: coil is well-captured (2340/2496 = 94%), but the model occasionally labels coil residues as H or E.

**Strand (E) is almost always the hardest class** — it depends more on long-range inter-strand hydrogen bonds than on local sequence patterns, and it is the minority class, so the model has seen fewer examples.

### Boundary errors

A **boundary error** is a misclassification that occurs at the first or last residue of a secondary-structure run, rather than in its interior. Boundary residues are harder because their window contains neighbours from *two* different classes simultaneously.

To measure this, label every residue as "boundary" (the first or last of a contiguous run) or "interior" and compare accuracy:

```python
def label_boundaries(ss_string):
    """Return a boolean array: True = boundary residue."""
    n = len(ss_string)
    boundary = [False] * n
    for i in range(n):
        left_diff  = (i == 0) or (ss_string[i] != ss_string[i - 1])
        right_diff = (i == n - 1) or (ss_string[i] != ss_string[i + 1])
        boundary[i] = left_diff or right_diff
    return boundary

# Compare Q3 accuracy on interior vs boundary positions
boundary_mask = np.array([b for ss in val_ss_strings for b in label_boundaries(ss)])
interior_mask = ~boundary_mask

q3_interior = (pred[interior_mask] == true[interior_mask]).mean()
q3_boundary = (pred[boundary_mask] == true[boundary_mask]).mean()
print(f"Interior Q3: {q3_interior:.3f}   Boundary Q3: {q3_boundary:.3f}")
```

Boundary accuracy is typically 5–15 percentage points lower than interior accuracy. If boundary errors dominate your confusion matrix, post-processing (see below) is a natural next step.

### Segment length effects

Short segments are harder to predict than long ones. A helix of three residues has almost the same sequence pattern as a coil turn; only a helix of seven or more residues has the canonical periodicity that one-hot and PSSM features can detect.

Plot accuracy as a function of **true segment length** to see this directly:

```python
from collections import defaultdict

def run_lengths_and_accuracy(true_labels, pred_labels):
    """Yield (class, run_length, fraction_correct) for each run in true_labels."""
    i = 0
    while i < len(true_labels):
        cls = true_labels[i]; j = i
        while j < len(true_labels) and true_labels[j] == cls:
            j += 1
        run_len = j - i
        correct = sum(pred_labels[k] == cls for k in range(i, j)) / run_len
        yield cls, run_len, correct
        i = j
```

### Per-protein error analysis

Reporting a single Q3 accuracy averaged over all residues hides a wide distribution. Compute Q3 accuracy **per protein** and look at the tail:

```python
per_protein_q3 = {}
for chain_id, group in val_df.groupby("protein_id"):
    t = group["true_ss"].values
    p = group["pred_ss"].values
    per_protein_q3[chain_id] = (t == p).mean()

# Hardest proteins
sorted_chains = sorted(per_protein_q3, key=per_protein_q3.get)
print("10 hardest proteins:", sorted_chains[:10])
```

Investigate the hardest proteins. Common patterns:
- **No PSSM signal** — the protein has few homologs, so PSI-BLAST found nothing and the PSSM is uninformative.
- **Membrane proteins** — transmembrane helices have unusual sequence statistics; the training set is biased toward soluble proteins.
- **Intrinsically disordered proteins** — large coil-rich regions with weak local signal.
- **Short proteins** — fewer residues means fewer training examples in that protein's style.

### Iteration strategies

| Symptom | Likely cause | What to try |
|---------|-------------|-------------|
| Both train and val Q3 are low | Underfitting | More expressive model, more features (ESM2), longer window |
| Train Q3 >> val Q3 | Overfitting | More regularisation, more training data, smaller model |
| Strand (E) accuracy much lower than H and C | Class imbalance / long-range dependency | Class-weighted loss, BiLSTM/Transformer to capture long range, oversample strand-rich chains |
| Boundary Q3 much lower than interior | Window too small; post-processing needed | Wider window, smoothing pass |
| Hard proteins have no homologs | PSSM uninformative | Fall back to ESM2 embeddings, which encode evolutionary information implicitly |

### Smoothing predictions (post-processing)

The simplest post-processing step: a **sliding-window majority vote** that removes implausibly short predicted segments. If a single predicted residue is surrounded by a different class on both sides, it is almost certainly a false positive:

```python
def smooth_predictions(pred_string, min_segment=3):
    """
    Remove runs shorter than min_segment by assigning them
    to the dominant class of their immediate neighbourhood.
    """
    labels = list(pred_string)
    changed = True
    while changed:
        changed = False
        i = 0
        while i < len(labels):
            cls = labels[i]; j = i
            while j < len(labels) and labels[j] == cls:
                j += 1
            if (j - i) < min_segment:
                # replace short run with neighbour class (left neighbour wins)
                replacement = labels[i - 1] if i > 0 else (labels[j] if j < len(labels) else cls)
                for k in range(i, j):
                    labels[k] = replacement
                changed = True
            i = j
    return "".join(labels)
```

Set `min_segment` to 3–4; the hydrogen-bond geometry of alpha helices requires at least 4 residues and beta strands at least 2, so predicted runs shorter than this are suspect.

---

## 3. Inferring Biological Properties

### From accuracy metric to biological insight

Until now you have measured your model's performance using Q3 accuracy — how often did it get the right label? In Week 6 we flip the question: **given a prediction, what can you learn about the protein?**

Secondary structure prediction was never just an ML benchmark. The output — a string of H, E, and C labels — encodes rich biological information about what a protein looks like and, indirectly, what it does.

### Secondary structure content as a protein fingerprint

The overall fractions of H, E, and C in a protein encode its **structural class**. This is one of the oldest and most robust features in structural biology:

| Structural class | Typical H% | Typical E% | Examples |
|-----------------|-----------|-----------|---------|
| **All-alpha** | > 50% | < 5% | Myoglobin, haemoglobin, cytochrome c |
| **All-beta** | < 5% | > 30% | Immunoglobulins, beta-barrels, lectins |
| **Alpha/beta** | 20–40% | 15–30% | TIM barrels, Rossmann folds, most enzymes |
| **Mostly coil / disordered** | < 15% | < 10% | Intrinsically disordered proteins (IDPs) |

A predicted secondary-structure string lets you calculate these fractions instantly, giving a first-pass structural classification of any protein — even one with no known homolog.

```python
def ss_composition(pred_string):
    n = len(pred_string)
    return {
        "H%": pred_string.count("H") / n * 100,
        "E%": pred_string.count("E") / n * 100,
        "C%": pred_string.count("C") / n * 100,
    }
```

### Identifying transmembrane helices

A stretch of **20 or more consecutive helix residues** in a protein with a hydrophobic sequence profile is a strong indicator of a **transmembrane (TM) helix** — a helix that spans the lipid bilayer. TM helices are typically 20–30 residues long and composed of hydrophobic amino acids (L, I, V, F, M, A).

```python
def find_long_runs(pred_string, label, min_len):
    """Return list of (start, end) for runs of `label` at least `min_len` long."""
    runs = []; i = 0
    while i < len(pred_string):
        if pred_string[i] == label:
            j = i
            while j < len(pred_string) and pred_string[j] == label:
                j += 1
            if j - i >= min_len:
                runs.append((i, j - 1))
            i = j
        else:
            i += 1
    return runs

long_helices = find_long_runs(pred_string, "H", min_len=20)
# Each entry is a candidate transmembrane helix
```

The number of predicted long helices gives a rough count of **transmembrane segments** — proteins with 7 TM helices are likely G-protein coupled receptors (GPCRs), the largest drug target family in the human genome.

### Intrinsically disordered regions (IDRs)

A **long continuous coil stretch** (> 30 residues of C) with no helix or strand often indicates an **intrinsically disordered region** — a segment of the protein that does not adopt a fixed 3D shape. IDRs are far from random: they often mediate protein-protein interactions through **short linear motifs (SLiMs)** buried within the disordered stretch, and they are disproportionately involved in signalling and regulation.

```python
disordered_regions = find_long_runs(pred_string, "C", min_len=30)
```

If a protein has many IDRs, its function likely involves dynamic, conditional interactions rather than a fixed enzymatic mechanism.

### Connecting predictions to functional annotations

The most powerful downstream analysis combines secondary-structure predictions with **UniProt functional annotations**. Load the UniProt features for your protein (active site, binding site, disulfide bond, etc.) and compare their positions to your predictions:

```python
# Pseudo-code — fetch UniProt features via the REST API
import requests

def get_uniprot_features(accession):
    url = f"https://rest.uniprot.org/uniprotkb/{accession}.json"
    data = requests.get(url).json()
    return data.get("features", [])

features = get_uniprot_features("P00760")   # bovine trypsin

for f in features:
    start = f["location"]["start"]["value"] - 1   # 0-indexed
    end   = f["location"]["end"]["value"]
    ftype = f["type"]
    predicted_ss = pred_string[start:end]
    print(f"{ftype} ({start+1}–{end}): predicted as '{set(predicted_ss)}'")
```

Typical findings:
- **Active site residues** are often in coil (C) loops adjacent to helices or strands — the flexible active-site loops position the catalytic residues.
- **Disulfide bonds** connect cysteine residues that are often in coil or turn positions.
- **Binding sites** can be in any class, but a position that is conserved (high PSSM magnitude) *and* predicted as coil is a strong candidate for a functional contact residue.

### Comparing to AlphaFold structures

[**AlphaFold2**](https://alphafold.ebi.ac.uk/) has predicted the structures of virtually all known proteins. You can download any AlphaFold structure, run DSSP on it (exactly as in Week 2), and compare the DSSP-derived labels to your model's predictions:

```python
from Bio.PDB import PDBParser
from Bio.PDB.DSSP import DSSP

def dssp_from_alphafold(pdb_path, chain="A"):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("af2", pdb_path)
    dssp = DSSP(structure[0], pdb_path)
    Q3_MAP = {"H":"H","G":"H","I":"H","E":"E","B":"E","T":"C","S":"C","C":"C","-":"C"}
    return "".join(Q3_MAP.get(v[2], "C") for (c, _), v in dssp.property_dict.items() if c == chain)

af2_ss  = dssp_from_alphafold("AF-P00760-F1-model_v4.pdb")
your_ss = pred_string

matches = sum(a == b for a, b in zip(af2_ss, your_ss))
print(f"Agreement with AlphaFold DSSP: {matches / len(af2_ss):.1%}")
```

This comparison is particularly useful for proteins without an experimental PDB structure — AlphaFold gives you a high-quality reference even when no crystal structure exists.

AlphaFold also outputs a **per-residue confidence score called pLDDT** (0–100). Residues with pLDDT < 50 are considered disordered by AlphaFold; your model should predict `C` at those positions, and if it does not, those are informative errors.

```python
def plddt_from_bfactor(pdb_path, chain="A"):
    """AlphaFold encodes pLDDT in the B-factor column."""
    from Bio.PDB import PDBParser
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("af2", pdb_path)
    scores = []
    for residue in structure[0][chain]:
        for atom in residue:
            if atom.get_name() == "CA":
                scores.append(atom.get_bfactor())
                break
    return scores   # one value per residue

plddt = plddt_from_bfactor("AF-P00760-F1-model_v4.pdb")
disordered_by_af2 = [i for i, s in enumerate(plddt) if s < 50]
```

### Where to go from here

Secondary structure prediction is a solved sub-problem — the tools you have built reach close to the ceiling of what sequence information can deliver. The frontier has moved:

| Next problem | What it requires | Tools |
|-------------|-----------------|-------|
| **Tertiary structure** | Predict the full 3D fold | AlphaFold2, RoseTTAFold, ESMFold |
| **Protein–protein interaction** | Which proteins bind each other, and where | AlphaFold-Multimer, RoseTTAFold2NA |
| **Function prediction** | GO terms, enzyme class, binding partners | ProteinBERT, ESM2 fine-tuned classifiers |
| **Protein design** | Invent new sequences that fold into a target structure | ProteinMPNN, RFdiffusion, ESM-IF |
| **Variant effect prediction** | How does a mutation change function? | EVE, ESM-1v, AlphaMissense |

Each of these builds directly on the same representation and modelling ideas you have used in this course — embeddings, per-residue classification heads, evolutionary information, and careful evaluation on held-out proteins. The tools change; the principles do not.

---

## Resources

- [AlphaFold database](https://alphafold.ebi.ac.uk/) — predicted structures for virtually all known proteins; download PDB files and compare DSSP labels to your predictions
- [UniProt REST API](https://www.uniprot.org/help/programmatic_access) — fetch functional annotations for any protein programmatically
- [PSIPRED server](https://bioinf.cs.ucl.ac.uk/psipred/) — the state-of-the-art classical secondary-structure predictor; compare your outputs to it
- [NetSurf-2.0](https://services.healthtech.dtu.dk/services/NetSurf-2.0/) — solvent accessibility and secondary structure; useful for understanding what your model is and is not capturing
- [SCOP — Structural Classification of Proteins](https://scop.mrc-lmb.cam.ac.uk/) — the canonical resource for protein structural classes (all-alpha, all-beta, alpha/beta, etc.)
- [CATH database](https://www.cathdb.info/) — another structural classification hierarchy, often used alongside SCOP
- [Jumper et al. (2021) — AlphaFold2 paper](https://www.nature.com/articles/s41586-021-03819-2) — the landmark result that effectively solved protein structure prediction
- [Biopython PDB tutorial](https://biopython.org/docs/latest/Tutorial/chapter_pdb.html) — parsing PDB files and running DSSP in Python
- [A review of protein secondary structure prediction methods (2022)](https://pmc.ncbi.nlm.nih.gov/articles/PMC9249596/) — covers classical through deep learning approaches

---

[← Home](../../README.md)
