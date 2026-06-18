# Week 3 — Feature Engineering & Classical Classification

[← Home](../../README.md) | [Week 3 Assignment →](assignment.md)

---

> **Building on Week 2.** You can now turn a protein structure into per-residue secondary-structure labels (H / E / C) and encode a sequence as a sliding-window one-hot matrix. This week we ask the central question of the whole course for the first time: *given only the sequence, can a machine predict the structure label?* You will build, tune, and compare the four pillars of classical machine learning to do exactly that.

---

## Topics

1. [Turning Sequences into Features — One-Hot, K-mers & TF-IDF](#1-turning-sequences-into-features)
2. [Probabilistic Classifiers — Naive Bayes & HMMs](#2-probabilistic-classifiers)
3. [Support Vector Machines — Maximum-Margin Classification](#3-support-vector-machines)
4. [Random Forests & Decision Trees](#4-random-forests--decision-trees)
5. [The Week 3 Task — Binary Secondary-Structure Prediction](#5-the-week-3-task)

---

## 1. Turning Sequences into Features

### The fundamental problem (recap)

Every classifier you will meet this week takes a **fixed-length vector of numbers** as input and produces a label. But a protein is a variable-length string of letters like `MNIFEMLRID...`. The entire art of classical ML on sequences is in the step *before* the model: how do you turn that string into numbers that carry the signal the model needs?

You already met one answer in Week 2 — the **sliding-window one-hot encoding**. This week we add two more — **k-mers** and **TF-IDF** — and you will compare them head-to-head in the assignment.

### 1a. One-hot encoding (recap from Week 2)

Each residue becomes a 20-dimensional binary vector (one slot per amino acid). To give a model *context*, we encode a window of `w` residues centred on the target residue and concatenate them:

```
full sequence:  ... A  G  K  [L  V  T  I  S  E]  R  P ...
                              |                 |
                         window of size 7 centred on I  -> 7 x 20 = 140 numbers
```

A window of size 17 gives `17 × 20 = 340` features per residue — the matrix of shape `(L, 340)` you built last week. This is our **baseline feature set**.

**What one-hot cannot do:** it treats every amino acid as equally different from every other (V and L look as unrelated as V and K), and it has no notion of *which short motifs* appear in the window — only which single residues sit at which positions.

→ [One-hot encoding — Wikipedia](https://en.wikipedia.org/wiki/One-hot)

### 1b. K-mers — the "words" of a sequence

A **[k-mer](https://en.wikipedia.org/wiki/K-mer)** is a contiguous subsequence of length `k`. Sliding a window of width `k` along a sequence and reading off every substring gives you that sequence's k-mers:

```
sequence:   M N I F E M L
1-mers:     M, N, I, F, E, M, L
2-mers:     MN, NI, IF, FE, EM, ML
3-mers:     MNI, NIF, IFE, FEM, EML
```

The big idea: **treat a protein sequence like a sentence and its k-mers like words.** Once you do that, every tool ever invented for text classification becomes available to you. A sequence is summarised by its **k-mer composition** — a vector counting how often each possible k-mer appears.

| `k` | Vocabulary size (20 amino acids) | What it captures |
|-----|----------------------------------|------------------|
| 1 | 20 | Amino-acid composition only |
| 2 | 400 | Adjacent-pair preferences (di-peptides) |
| 3 | 8,000 | Short local motifs |
| 4 | 160,000 | Longer motifs — usually too sparse for classical ML |

Notice the trade-off: larger `k` captures richer patterns but the vector explodes in size and most counts become zero (**sparsity**). For per-residue prediction in this course, `k = 1` to `3` over a window is the sweet spot.

**Per-residue twist.** K-mer composition naturally describes a *whole* sequence, but our task labels *each residue*. The fix: treat the **window around a residue as a mini-document** and compute the k-mer composition of that window. Slide the window and you get one k-mer vector per residue — directly comparable to the one-hot matrix.

```python
def kmers(seq, k=3):
    return [seq[i:i + k] for i in range(len(seq) - k + 1)]

# scikit-learn can vectorise k-mers for you with a custom analyzer
from sklearn.feature_extraction.text import CountVectorizer

vec = CountVectorizer(analyzer=lambda s: kmers(s, k=3))
window_strings = ["LVTISER", "VTISERP", ...]   # one window per residue
X_counts = vec.fit_transform(window_strings)    # sparse (n_residues, vocab)
```

→ [K-mer — Wikipedia](https://en.wikipedia.org/wiki/K-mer)
→ [scikit-learn CountVectorizer docs](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html)

### 1c. TF-IDF — weighting k-mers by how informative they are

Raw k-mer counts have a problem familiar from text: the most *frequent* terms are often the least *useful*. A k-mer that appears in almost every window (because that residue pair is common everywhere) tells you nothing about whether *this* residue is in a helix.

**[TF-IDF](https://en.wikipedia.org/wiki/Tf%E2%80%93idf)** (Term Frequency × Inverse Document Frequency) fixes this. For a k-mer `t` in a window `d` drawn from a corpus of `N` windows:

```
tf(t, d)   = count of t in d  (optionally normalised by window length)
idf(t)     = log( N / number of windows containing t )
tfidf(t,d) = tf(t, d) × idf(t)
```

- A k-mer that appears **everywhere** has a small `idf` → downweighted.
- A k-mer that appears in **few** windows but is present here has a large `idf` → upweighted.

The effect: TF-IDF automatically emphasises the **discriminative** motifs and mutes the generic ones, often improving classical classifiers with no extra modelling effort.

```python
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(analyzer=lambda s: kmers(s, k=3))
X_tfidf = tfidf.fit_transform(window_strings)   # same shape, weighted values
```

→ [TF-IDF — Wikipedia](https://en.wikipedia.org/wiki/Tf%E2%80%93idf)
→ [scikit-learn TfidfVectorizer docs](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)

### Which feature set wins?

You do not get to guess — you measure. Part of this week's assignment is to run the **same model** on one-hot, k-mer-count, and TF-IDF features and report which gives the best validation F1. This is the everyday reality of classical ML: the feature engineering usually matters more than the choice of classifier.

> **Looking ahead to Week 4.** Every feature in this section is *hand-crafted* — you, the human, decided that windows, k-mers, and TF-IDF weighting are the right summary. Next week, ESM2 / ProtT5 embeddings will *learn* the representation from hundreds of millions of sequences, and you will see how much head-room hand-crafted features leave on the table.

---

## 2. Probabilistic Classifiers

The next two models are **generative / probabilistic**: instead of drawing a boundary between classes, they model *how each class generates its data* and then ask which class most probably produced the residue in front of them.

### 2a. Naive Bayes

#### The idea

[Naive Bayes](https://en.wikipedia.org/wiki/Naive_Bayes_classifier) applies [Bayes' theorem](https://en.wikipedia.org/wiki/Bayes%27_theorem) with one bold simplifying assumption. We want the probability that a residue is a helix given its features `x = (x₁, ..., x_d)`:

```
P(helix | x)  ∝  P(helix) × P(x | helix)
```

The "naive" assumption is that all features are **conditionally independent** given the class:

```
P(x | helix) = P(x₁ | helix) × P(x₂ | helix) × ... × P(x_d | helix)
```

This is obviously false for proteins (neighbouring residues are correlated), yet Naive Bayes works *remarkably* well in practice — it is the classic baseline for text classification, and our k-mer features make the analogy exact.

#### Which variant to use

| Variant | Use when features are | Good fit here for |
|---------|----------------------|-------------------|
| **MultinomialNB** | Counts (non-negative) | K-mer count vectors |
| **BernoulliNB** | Binary present/absent | One-hot windows |
| **GaussianNB** | Continuous real values | TF-IDF (roughly) |

```python
from sklearn.naive_bayes import MultinomialNB

clf = MultinomialNB(alpha=1.0)   # alpha = Laplace smoothing (tune this!)
clf.fit(X_train, y_train)
preds = clf.predict(X_test)
```

The `alpha` smoothing parameter matters: it prevents a single never-before-seen k-mer from zeroing out an entire probability. Tuning it is part of the assignment.

→ [StatQuest: Naive Bayes — YouTube](https://www.youtube.com/watch?v=O2L2Uv9pdDA)
→ [Stanford CS229: Generative Learning notes](https://cs229.stanford.edu/notes2022/fall2022/cs229-notes2.pdf)
→ [scikit-learn Naive Bayes docs](https://scikit-learn.org/stable/modules/naive_bayes.html)

### 2b. Hidden Markov Models

#### Why a sequence model?

Naive Bayes labels each residue **in isolation**. But secondary structure is not scattered randomly — **helices and strands come in runs**. If residue `i` is in a helix, residue `i+1` is very likely a helix too. A [Hidden Markov Model](https://en.wikipedia.org/wiki/Hidden_Markov_model) is the classic way to bake that fact into the model.

#### The setup

An HMM imagines that an unseen "machine" walks along the protein, occupying one **hidden state** at each position and emitting one **observation** (the amino acid):

```
hidden states:   non-helix → non-helix → helix → helix → helix → non-helix
                     |           |          |       |       |        |
emissions:           M           N          I       F       E        M
```

The model is defined by three tables:

| Table | Symbol | Meaning |
|-------|--------|---------|
| **Start** | π | Probability the chain starts in each state |
| **Transition** | A | P(next state \| current state) — captures "helices come in runs" |
| **Emission** | B | P(amino acid \| state) — captures helix-preferring residues (A, E, L, M...) |

Because our labels are *known* at training time, you can estimate all three tables by **simple counting** (maximum likelihood) — no iterative training needed:

```
A[helix → helix]      = (# helix-to-helix transitions) / (# transitions out of helix)
B[helix emits 'A']    = (# alanines labelled helix) / (# residues labelled helix)
```

At test time, the **[Viterbi algorithm](https://en.wikipedia.org/wiki/Viterbi_algorithm)** finds the single most probable hidden-state path through a sequence — that path *is* your per-residue prediction.

```python
import numpy as np
from hmmlearn import hmm

model = hmm.CategoricalHMM(n_components=2, init_params="")
model.startprob_    = pi            # shape (2,)      estimated by counting
model.transmat_     = A             # shape (2, 2)
model.emissionprob_ = B             # shape (2, 20)
# encode each test sequence as integer amino-acid indices, then:
states = model.predict(obs_seq.reshape(-1, 1))   # Viterbi decoding -> 0/1 labels
```

> **Naive Bayes vs HMM in one line:** Naive Bayes asks *"what does this residue look like?"*; the HMM also asks *"what were its neighbours, and do labels tend to stick together?"* The HMM is the natural **probabilistic sequence-labelling** counterpart of Naive Bayes.

→ [StatQuest: Hidden Markov Models — YouTube](https://www.youtube.com/watch?v=fX5bYmrHzbY)
→ [Rabiner's classic HMM tutorial (1989)](https://www.ece.ucsb.edu/Faculty/Rabiner/ece259/Reprints/tutorial%20on%20hmm%20and%20applications.pdf)
→ [hmmlearn documentation](https://hmmlearn.readthedocs.io/)

---

## 3. Support Vector Machines

We now switch from generative to **discriminative** models — these draw a boundary between classes directly, without modelling how each class is generated.

### The maximum-margin idea

When two classes are separable, infinitely many lines can divide them. A [Support Vector Machine](https://en.wikipedia.org/wiki/Support_vector_machine) picks the one that leaves the **widest gap (margin)** between the classes. The handful of points sitting right on the margin are the **support vectors** — they alone define the boundary; everything else is irrelevant.

```
   non-helix  o   o            |  margin  |          x   x  helix
              o      o      o  | <------> | x   x
                  o            |          |     x
                          support vectors sit on the dashed edges
```

A wide margin tends to generalise better to unseen data — that is the whole intuition.

### The kernel trick

Real data is rarely separable by a straight line. SVMs handle this with the **[kernel trick](https://en.wikipedia.org/wiki/Kernel_method)**: implicitly map the features into a much higher-dimensional space where a linear boundary *does* exist, without ever computing that space explicitly.

| Kernel | When to use |
|--------|-------------|
| **Linear** | High-dimensional sparse features (k-mers, TF-IDF) — usually the right choice here, and fast |
| **RBF (Gaussian)** | Lower-dimensional dense features, non-linear boundaries — powerful but slower |

> **Practical note for this week.** With tens of thousands of training residues and high-dimensional sparse features, `LinearSVC` (or `SGDClassifier` with hinge loss) trains in seconds, while a full RBF `SVC` can be painfully slow. Start linear.

```python
from sklearn.svm import LinearSVC

clf = LinearSVC(C=1.0)   # C controls the margin/error trade-off — tune it
clf.fit(X_train, y_train)
```

The `C` hyperparameter trades off a wider margin against misclassifying training points: small `C` = wider, softer margin (more regularisation); large `C` = narrower, stricter margin (risk of overfitting). Tuning `C` is part of the assignment.

→ [StatQuest: Support Vector Machines — YouTube](https://www.youtube.com/watch?v=efR1C6CvhmE)
→ [Stanford CS229: SVM notes](https://cs229.stanford.edu/notes2022/fall2022/cs229-notes3.pdf)
→ [scikit-learn SVM docs](https://scikit-learn.org/stable/modules/svm.html)

---

## 4. Random Forests & Decision Trees

### Decision trees

A [decision tree](https://en.wikipedia.org/wiki/Decision_tree_learning) classifies by asking a sequence of yes/no questions about the features, splitting the data at each node to make the resulting groups as "pure" (single-class) as possible:

```
                 Is position 0 of window = 'A'?
                  /                          \
                yes                           no
                /                              \
       Is position +1 = 'L'?            Is position -1 = 'P'?
        /          \                       /          \
     HELIX      NON-HELIX             NON-HELIX      HELIX
```

Splits are chosen to maximise the drop in impurity, measured by **[Gini impurity](https://en.wikipedia.org/wiki/Decision_tree_learning#Gini_impurity)** or **entropy / information gain**. Trees are wonderfully interpretable but, left unrestrained, they memorise the training set — they **overfit**.

### Random forests — wisdom of many trees

A [Random Forest](https://en.wikipedia.org/wiki/Random_forest) tames overfitting by training many trees on random variations of the data and averaging their votes. Two sources of randomness keep the trees **decorrelated**:

1. **Bagging** — each tree trains on a bootstrap sample (random rows drawn with replacement).
2. **Feature subsampling** — at each split, each tree may only consider a random subset of features.

No single tree sees the whole picture, so their errors are independent and cancel out when averaged — a textbook example of **ensemble learning** reducing variance.

```python
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(
    n_estimators=300,      # number of trees
    max_depth=None,        # tune to control overfitting
    max_features="sqrt",   # features considered per split
    class_weight="balanced",  # helps with the helix/non-helix imbalance
    n_jobs=-1,
)
clf.fit(X_train, y_train)
```

A bonus you should use in your write-up: `clf.feature_importances_` tells you **which window positions and k-mers the forest relied on most** — a window of interpretability into what the model learned.

→ [StatQuest: Random Forests — YouTube](https://www.youtube.com/watch?v=J4Wdy0Wc_xQ)
→ [StatQuest: Decision Trees — YouTube](https://www.youtube.com/watch?v=_L39rN6gz7Y)
→ [scikit-learn ensembles docs](https://scikit-learn.org/stable/modules/ensemble.html)

### Generative vs discriminative — the map of this week

| Family | Model | Asks |
|--------|-------|------|
| **Generative / probabilistic** | Naive Bayes | "How likely is a helix to *produce* these features?" |
| | Hidden Markov Model | "What is the most probable label *path* along the chain?" |
| **Discriminative** | SVM | "Where is the widest *boundary* between the classes?" |
| | Random Forest / Tree | "What sequence of *questions* best separates them?" |

The assignment asks you to field **at least one from each family** and compare — because on real data, which family wins is an empirical question, not a theoretical one.

---

## 5. The Week 3 Task

### The prediction problem

This week is the course's **first true supervised-learning task**. The problem, stated precisely:

> Given a protein's amino-acid sequence, predict for **every residue** whether it sits in an **α-helix (label 1)** or **not (label 0)**.

This is the Week 2 Q3 labelling, collapsed to a clean binary: `H → 1`, and `E`/`C → 0`. Classical models thrive on binary problems; Week 4 will return to the full 3-class Q3 with deep learning.

You will train and tune several classical models, evaluate them on a **held-out split of proteins you set aside yourself**, and report the results — there is no separate hidden test set this week; the discipline of building an honest evaluation is part of what you are assessed on.

### Why F1, not just accuracy

Only about a third of residues are helices. A lazy model that predicts **"non-helix" for everything** scores ~65% accuracy while being completely useless. The **[F1-score](https://en.wikipedia.org/wiki/F-score)** — the harmonic mean of precision and recall on the helix class — punishes exactly that failure, which is why it is the metric you should lead with.

```
precision = TP / (TP + FP)      recall = TP / (TP + FN)
F1        = 2 · (precision · recall) / (precision + recall)
```

We report **macro-F1** (averaged over both classes) so that getting the minority class right actually counts.

### The one mistake that will wreck your results: data leakage

Residues from the *same protein* are highly correlated. If you split your data **by residue**, near-identical windows from one protein end up in both train and validation, and your reported F1 looks fantastic while meaning nothing — the model has effectively seen the answers.

**Always split by protein chain, never by residue.** Use `GroupShuffleSplit` or `GroupKFold` with the protein ID as the group. This single discipline separates an honest evaluation from a self-deceiving one, and the rubric rewards it.

```python
from sklearn.model_selection import GroupShuffleSplit

gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, val_idx = next(gss.split(X, y, groups=protein_ids))
```

### The pipeline you will build

```
labelled dataset (sequence + H/E/C labels)
        |
        |  collapse labels: H -> 1, E/C -> 0
        v
  per-residue windows  --> [ one-hot | k-mer counts | TF-IDF ]   (choose & compare)
        |
        v
  train & tune:  Naive Bayes / HMM        (>=1 probabilistic)
                 SVM / Random Forest       (>=1 discriminative)
        |
        |  group-aware split (split by protein!)
        v
  evaluate on held-out proteins  -->  report macro-F1, confusion matrix, comparison
```

Everything you need to do this is specified in the [assignment](assignment.md).

---

## Resources

- [scikit-learn user guide](https://scikit-learn.org/stable/user_guide.html) — the reference for every classifier, vectoriser, and metric used this week
- [StatQuest: Support Vector Machines](https://www.youtube.com/watch?v=efR1C6CvhmE) and [Random Forests](https://www.youtube.com/watch?v=J4Wdy0Wc_xQ) — intuition-first video walkthroughs
- [Stanford CS229: Generative Learning notes](https://cs229.stanford.edu/notes2022/fall2022/cs229-notes2.pdf) & [SVM notes](https://cs229.stanford.edu/notes2022/fall2022/cs229-notes3.pdf) — the rigorous treatment
- [hmmlearn documentation](https://hmmlearn.readthedocs.io/) — HMMs in Python
- [Rabiner's HMM tutorial (1989)](https://www.ece.ucsb.edu/Faculty/Rabiner/ece259/Reprints/tutorial%20on%20hmm%20and%20applications.pdf) — the canonical reference
- [A comparison of ML methods for protein secondary structure prediction (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC9249596/) — classical ML applied to this exact problem
- [Protein Secondary Structure dataset (RCSB Kabsch–Sander, curated CSV)](https://www.kaggle.com/datasets/alfrandom/protein-secondary-structure) — the labelled dataset used for this week's task

---

[← Home](../../README.md) | [Week 3 Assignment →](assignment.md)
