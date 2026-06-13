# Week 2 — Protein Structure, PDB & Sequence Representations

[← Home](../../README.md)

---

> **Building on Week 1.** You now know that a protein is a sequence of amino acids. This week we ask: what does that sequence actually *become* in 3D space, how do we store that 3D information, and how do we turn a sequence into numbers a computer can learn from?

---

## Topics

1. [Protein Structure — From Sequence to Shape](#1-protein-structure)
2. [The Protein Data Bank — 3D Structures at Your Fingertips](#2-the-protein-data-bank)
3. [DSSP — Labelling Secondary Structure Automatically](#3-dssp)
4. [Representing Sequences as Numbers — Sliding Window & Protein Language Models](#4-representing-sequences-as-numbers)

---

## 1. Protein Structure

### Why does shape matter?

A protein's job depends almost entirely on its **3D shape**. The sequence of amino acids you saw in Week 1 is just a blueprint — the real object is a folded, three-dimensional molecule. Two proteins with completely different sequences can do the same job if they fold into the same shape; conversely, a single mutation in the wrong place can destroy a protein's function by disrupting its shape.

Bioinformatics needs to deal with structure because, ultimately, **what we want to predict in this course — secondary structure — is a property of the 3D shape**, not the raw sequence.

### The four levels of protein structure

Biologists describe protein structure at four levels of organisation:

| Level | What it is | Example |
|-------|-----------|---------|
| **Primary** | The amino acid sequence itself | `MNIFEMLRID...` |
| **Secondary** | Local, repeating folding patterns formed by nearby residues | Alpha helices, beta sheets |
| **Tertiary** | The full 3D fold of a single protein chain | The globular shape of haemoglobin's alpha chain |
| **Quaternary** | Multiple chains assembled together | Full haemoglobin (2 alpha + 2 beta chains) |

→ [Khan Academy: Protein structure](https://www.khanacademy.org/science/ap-biology/gene-expression-and-regulation/translation/a/orders-of-protein-structure)

### Secondary structure — the patterns we will predict

Secondary structure is formed when backbone atoms in nearby residues form [hydrogen bonds](https://en.wikipedia.org/wiki/Hydrogen_bond) with each other. The two most important patterns are:

#### Alpha helix

Residues coil into a right-handed spiral. The backbone hydrogen-bonds every 4 residues along the chain. Helices are common in membrane proteins and in the interior of globular proteins.

```
        <- axis
   ...
   |  \
  N-H...O=C      (hydrogen bond between residue i and residue i+4)
   |
  C=O...H-N
   |
  ...
```

→ [Alpha helix — Wikipedia](https://en.wikipedia.org/wiki/Alpha_helix)

#### Beta sheet

Strands of the protein chain line up side by side, with hydrogen bonds running *between* strands rather than along one chain. Can be parallel (strands run in the same direction) or antiparallel.

→ [Beta sheet — Wikipedia](https://en.wikipedia.org/wiki/Beta_sheet)

#### Coil / loop

Any region that is not a helix or sheet. Often called a **random coil**, though it is not random — it has a defined shape, just no repeating hydrogen-bond pattern. Loops frequently form active sites and binding regions because they are flexible.

### The 3-class labelling (Q3)

Most secondary structure prediction work simplifies the 8 DSSP classes (see Section 3) into three broad categories:

| Label | Meaning | Includes |
|-------|---------|---------|
| **H** | Helix | Alpha helix, 3-10 helix, pi helix |
| **E** | Strand (extended) | Beta strand, isolated beta bridge |
| **C** | Coil | Turns, bends, everything else |

Prediction accuracy on this 3-class problem is called **Q3 accuracy**. State-of-the-art models reach ~85–90% Q3 on standard benchmarks.

---

## 2. The Protein Data Bank

### What is the PDB?

The [**Protein Data Bank (PDB)**](https://www.rcsb.org/) is the global archive for experimentally determined 3D structures of proteins, nucleic acids, and their complexes. It currently holds over **220,000 structures**.

Structures are determined experimentally using:
- **X-ray crystallography** — most common; protein is crystallised and hit with X-rays
- **Cryo-electron microscopy (Cryo-EM)** — rapidly growing; protein is flash-frozen and imaged with electrons
- **NMR spectroscopy** — gives structures in solution; best for smaller proteins

Every structure has a **PDB ID** — a 4-character alphanumeric code, e.g., `1TGT` (bovine trypsin).

→ [Search the PDB](https://www.rcsb.org/)
→ [RCSB PDB beginner tutorial](https://www.rcsb.org/pages/educational_resources)

### Reading a PDB entry

Open any entry at `https://www.rcsb.org/structure/<PDBID>`. Key sections to look at:

| Section | What it tells you |
|---------|-----------------|
| **Summary** | Organism, experimental method, resolution, deposition date |
| **Macromolecules** | Chain IDs, sequence, linked UniProt entry |
| **3D View** | Interactive visualisation in the browser |
| **Download** | `.pdb` or `.cif` file for local analysis |

### The PDB file format

A `.pdb` file is a fixed-width text file. The most important record type is `ATOM`:

```
ATOM      1  N   ILE A  16      16.756  20.346   7.629  1.00 12.45           N
ATOM      2  CA  ILE A  16      17.589  19.208   7.241  1.00 11.60           C
```

| Column | Meaning |
|--------|---------|
| `ATOM` | Record type (ATOM = standard residue; HETATM = ligand/water) |
| `ILE` | Residue name (3-letter code) |
| `A` | Chain ID |
| `16` | Residue sequence number |
| `16.756 20.346 7.629` | X, Y, Z coordinates in Angstroms |
| `12.45` | B-factor (thermal motion — higher means more flexible) |

→ [Full PDB format specification](https://www.wwpdb.org/documentation/file-format)

### Fetching and parsing PDB files with Biopython

    from Bio.PDB import PDBParser, PDBList

    # Download a structure from the PDB
    pdbl = PDBList()
    pdbl.retrieve_pdb_file("1TGT", file_type="pdb", pdir=".")

    # Parse it
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("trypsin", "pdb1tgt.ent")

    # Iterate over chains and residues
    for model in structure:
        for chain in model:
            for residue in chain:
                print(residue.resname, residue.id[1])

→ [Biopython PDB module docs](https://biopython.org/docs/latest/Tutorial/chapter_pdb.html)

### Visualising structures

You do not need to install anything to see a structure — the PDB website has a built-in viewer. For more control:

| Tool | Type | Best for |
|------|------|---------|
| [NGL Viewer](https://nglviewer.org/) | Browser — no install | Quick interactive viewing |
| [PyMOL](https://pymol.org/) | Desktop | Publication-quality figures |
| [ChimeraX](https://www.cgl.ucsf.edu/chimerax/) | Desktop | Cryo-EM, large complexes |
| [py3Dmol](https://github.com/3dmol/3Dmol.js) | Python / Jupyter | Inline notebook viewing |

---

## 3. DSSP

### What is DSSP?

[**DSSP**](https://en.wikipedia.org/wiki/DSSP_(hydrogen_bond_estimation_algorithm)) stands for **Define Secondary Structure of Proteins**. It is an algorithm published by [Kabsch and Sander (1983)](https://onlinelibrary.wiley.com/doi/10.1002/bip.360221211) that reads a PDB file and assigns a secondary structure label to **every single residue** based on the hydrogen-bond geometry of the backbone atoms.

This is crucial for our course: **DSSP generates the ground-truth labels we want our ML model to predict.** The pipeline is:

```
PDB structure  --(DSSP)--> per-residue labels (H / E / C)
Sequence alone --(model)--> per-residue predictions
```

Training a model means teaching it to reproduce DSSP labels using only the sequence as input — without ever seeing the 3D structure.

### The 8 DSSP classes

DSSP assigns one of 8 labels to each residue:

| Label | Name | Description |
|-------|------|-------------|
| `H` | Alpha helix | Standard right-handed helix |
| `G` | 3-10 helix | Tighter helix, 3 residues per turn |
| `I` | Pi helix | Rare, wider helix |
| `E` | Beta strand | Part of a beta sheet |
| `B` | Isolated beta bridge | Single pair of hydrogen bonds between two residues |
| `T` | Turn | Hydrogen-bonded turn (4 types) |
| `S` | Bend | Geometric bend with no hydrogen bond |
| `C` | Coil | Everything else |

For most ML work, these 8 labels are collapsed into the 3-class Q3 scheme from Section 1.

### Running DSSP with Biopython

    from Bio.PDB import PDBParser
    from Bio.PDB.DSSP import DSSP

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("trypsin", "pdb1tgt.ent")
    model = structure[0]

    dssp = DSSP(model, "pdb1tgt.ent")

    for (chain_id, res_id), values in dssp.property_dict.items():
        residue_name = values[1]   # amino acid one-letter code
        ss_label     = values[2]   # DSSP secondary structure label
        print(chain_id, res_id[1], residue_name, ss_label)

> **Note**: DSSP must be installed separately. On Linux/Mac: `conda install -c conda-forge dssp`. On Windows, use [mkdssp](https://github.com/PDB-REDO/dssp).

→ [Biopython DSSP docs](https://biopython.org/docs/latest/Tutorial/chapter_pdb.html#dssp)
→ [Original DSSP paper — Kabsch & Sander (1983)](https://onlinelibrary.wiley.com/doi/10.1002/bip.360221211)

---

## 4. Representing Sequences as Numbers

### The fundamental problem

Machine learning models work with numbers, not letters. Given a sequence like `MNIFEMLRID...`, we need to convert it into a matrix or vector before any model can process it. This section covers two approaches: a **classical** hand-crafted one and a **modern** learned one.

---

### 4a. Sliding Window Encoding

#### The idea

Instead of encoding the entire sequence at once, we look at each residue **in the context of its neighbours**. For each residue at position *i*, we take a window of *w* residues centred on *i*:

```
full sequence:  ... A  G  K  [L  V  T  I  S  E]  R  P ...
                              |                 |
                         window of size 7 centred on I
```

We then encode this window as a single feature vector, and use it to predict the secondary structure label at position *i*. Sliding the window along every position produces one feature vector per residue.

#### One-hot encoding

The simplest way to encode a residue is **one-hot encoding**: represent each amino acid as a 20-dimensional binary vector where only the position for that amino acid is 1, all others are 0.

```
A -> [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
C -> [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
...
```

→ [One-hot encoding — Wikipedia](https://en.wikipedia.org/wiki/One-hot)

#### Building the feature matrix

For a window of size 17 (8 residues on each side), each position becomes a vector of 17 x 20 = **340 numbers**. A protein of length *L* becomes a matrix of shape **(L, 340)**.

    import numpy as np

    AA_ORDER = "ACDEFGHIKLMNPQRSTVWY"
    aa_to_idx = {aa: i for i, aa in enumerate(AA_ORDER)}

    def one_hot(aa):
        vec = np.zeros(20)
        if aa in aa_to_idx:
            vec[aa_to_idx[aa]] = 1.0
        return vec

    def sliding_window_encode(sequence, window=17):
        pad = window // 2
        padded = "-" * pad + sequence + "-" * pad
        features = []
        for i in range(len(sequence)):
            window_seq = padded[i : i + window]
            features.append(np.concatenate([one_hot(aa) for aa in window_seq]))
        return np.array(features)   # shape: (L, window * 20)

#### Limitations

- Treats each window independently — no long-range context beyond the window.
- Nothing in one-hot encoding tells the model that V and L are chemically similar.
- Works reasonably well with classical ML (Week 3) but hits a performance ceiling.

---

### 4b. Protein Language Models (PLMs)

#### What is a protein language model?

A [**protein language model**](https://en.wikipedia.org/wiki/Protein_language_model) is a deep neural network — usually a [Transformer](https://en.wikipedia.org/wiki/Transformer_(deep_learning_architecture)) — trained on hundreds of millions of protein sequences from databases like [UniRef](https://www.uniprot.org/help/uniref). Just as GPT learns the grammar and meaning of text, a PLM learns the "grammar" of protein sequences: which amino acids tend to co-occur, which patterns are conserved, and implicitly, what the sequence means structurally.

The key output is a **per-residue embedding**: for each amino acid in a sequence, the model produces a dense vector (typically 480–1280 numbers) that encodes rich contextual information about that residue — including information from far away in the sequence.

#### Why are PLM embeddings better than one-hot?

| Property | One-hot | PLM embedding |
|----------|---------|---------------|
| Captures chemical similarity | No | Yes |
| Encodes long-range context | No (window only) | Yes (full sequence) |
| Contains evolutionary information | No | Yes |
| Dimensionality per residue | 20 | 480–1280 |
| Requires pretraining | No | Yes (done for you) |

#### ESM2 — the most widely used PLM

[**ESM2**](https://github.com/facebookresearch/esm) (Evolutionary Scale Modeling) is developed by Meta AI Research. It comes in several sizes; `esm2_t6_8M_UR50D` is the smallest and fastest to experiment with.

    # pip install fair-esm torch
    import esm, torch

    model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
    batch_converter = alphabet.get_batch_converter()
    model.eval()

    data = [("protein1", "MNIFEMLRIDEKLGLGTSFPVHLT")]
    batch_labels, batch_strs, batch_tokens = batch_converter(data)

    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[6])
    embeddings = results["representations"][6]
    # Shape: (1, sequence_length + 2, 320)  -- +2 for start/end tokens

→ [ESM2 GitHub repository](https://github.com/facebookresearch/esm)
→ [ESM2 paper — Lin et al. (2022)](https://www.science.org/doi/10.1126/science.ade2574)

#### ProtTrans — another strong PLM family

[**ProtTrans**](https://github.com/agemagician/ProtTrans) by the Rost Lab offers several models (ProtBERT, ProtT5) trained on UniRef and BFD databases. ProtT5 gives excellent embeddings for secondary structure prediction tasks.

    # pip install transformers torch sentencepiece
    from transformers import T5Tokenizer, T5EncoderModel
    import torch

    tokenizer = T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_half_uniref50-enc")
    model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_half_uniref50-enc")
    model.eval()

    sequence = "MNIFEMLRIDEKLGLGTSFPVHLT"
    spaced = " ".join(list(sequence))   # ProtTrans expects space-separated residues
    ids = tokenizer(spaced, return_tensors="pt")

    with torch.no_grad():
        embeddings = model(**ids).last_hidden_state
    # Shape: (1, sequence_length, 1024)

→ [ProtTrans GitHub repository](https://github.com/agemagician/ProtTrans)
→ [ProtTrans paper — Elnaggar et al. (2021)](https://ieeexplore.ieee.org/document/9477085)

#### Connecting it all: what we will build

By the end of this course you will combine everything from Weeks 1–5:

```
Protein sequence
       |
       |---> Sliding window one-hot    ---> Week 3 classical ML models
       |
       `--> ESM2 / ProtT5 embeddings  ---> Week 4 deep learning models
                                                   |
                                                   v
                                       Per-residue H / E / C prediction
                                       (evaluated against DSSP labels)
```

---

## Resources

- [RCSB Protein Data Bank](https://www.rcsb.org/) — search, visualise, and download protein structures
- [PDB-101: Introduction to protein structure](https://pdb101.rcsb.org/learn/guide-to-understanding-pdb-data/introduction) — gentle visual introduction
- [DSSP web server](https://www3.cmbi.umcn.nl/dssp/) — run DSSP on any PDB file online, no installation needed
- [Kabsch & Sander (1983)](https://onlinelibrary.wiley.com/doi/10.1002/bip.360221211) — original DSSP paper
- [ESM2 repository](https://github.com/facebookresearch/esm) — Meta AI protein language models
- [ProtTrans repository](https://github.com/agemagician/ProtTrans) — ProtBERT, ProtT5 models
- [Biopython PDB tutorial](https://biopython.org/docs/latest/Tutorial/chapter_pdb.html) — parsing and analysing PDB files in Python

---

[← Home](../../README.md) | [Assignment →](assignment.md)
