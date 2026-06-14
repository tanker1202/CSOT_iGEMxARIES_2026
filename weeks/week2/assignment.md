# Week 2 Assignment — PDB Parsing, DSSP Labelling & Sliding Window Encoding

[← Week 2 Topics](index.md) | [Leaderboard →](../../leaderboard.md)

---

## Task

Choose a protein from the table below, download its structure in mmCIF format, extract per-residue secondary structure labels using `pydssp`, and build a sliding window one-hot feature matrix. Submit a report covering all three deliverables.

> This assignment is the foundation for everything ahead: the Q3 labels you extract here are the **ground-truth targets** your ML model will learn to predict in Weeks 3 and 4.

### Suggested proteins (pick one, or choose your own)

| Option | PDB ID | Protein | Why interesting |
|--------|--------|---------|-----------------|
| A | [1TGT](https://www.rcsb.org/structure/1TGT) | Bovine Trypsin | Familiar from Week 1; clear mix of helix, strand, and coil |
| B | [2LYZ](https://www.rcsb.org/structure/2LYZ) | Hen Egg-White Lysozyme | Classic teaching protein; well-annotated, mixed secondary structure |
| C | [1MBN](https://www.rcsb.org/structure/1MBN) | Sperm Whale Myoglobin | Predominantly alpha-helical; great example of Q3 class imbalance |
| D | [1UBQ](https://www.rcsb.org/structure/1UBQ) | Human Ubiquitin | Short (76 residues), fast to run; clean mix of helix, strand, and coil |

You may choose your own protein — it must have a solved X-ray or cryo-EM structure in the PDB, and you must justify the choice in your report.

---

## Deliverables

### 1. Structure Summary

Parse the mmCIF file using Biopython and report the following for **one chain of your choice**:

- PDB ID, protein name, organism, experimental method, and resolution (find these on the RCSB page or in the PDB header).
- Chain ID you analysed and total number of residues in that chain.
- A table of the first **5 ATOM records** for the Cα atoms (atom name `CA`) in your chosen chain, showing: residue name, residue number, and X/Y/Z coordinates.

> Tip: filter for `atom.get_name() == "CA"` when iterating over atoms.

### 2. Secondary Structure Labelling

Run secondary structure assignment on your structure using `pydssp` and report:

- The **Q3 distribution** — count and percentage of residues in each of the three classes: H (helix), E (strand), C (coil).
- The **longest continuous helix** (Q3 label `H`) and the **longest continuous strand** (Q3 label `E`) in your chain — give the start and end residue numbers and the length of each run.
- A text or ASCII representation of the per-residue Q3 label string across the full chain (just print the string of H/E/C characters).

### 3. Sliding Window Feature Matrix

Implement the one-hot sliding window encoding from the Week 2 notes (window size = 17, pad with `-` at the chain boundaries) and report:

- The shape of the resulting feature matrix `(L, 340)` where `L` is the chain length.
- For residues assigned Q3 label `H` and residues assigned Q3 label `E`, compute and compare the **mean one-hot frequency** for each of the 20 amino acids across all windows centred on residues of that class. Report the top 3 most enriched amino acids in H-windows and in E-windows (i.e., which amino acids appear most often in windows centred on helix vs. strand residues).
- Write **2–4 sentences** interpreting the result: do the enriched amino acids match known helix-forming or strand-forming tendencies? (You may look up the Chou-Fasman parameters or any secondary structure propensity scale to support your answer.)

---

## Recommended Workflow

```
1. Download the mmCIF file
   - Use PDBList (shown in the code snippets below) — saves as <id>.cif

2. Parse the structure
   - Use Bio.PDB.MMCIFParser to load the file
   - Pick a single chain to analyse

3. Assign secondary structure
   - Extract backbone coordinates (N, CA, C, O) from your chain
   - Run pydssp.assign() to get per-residue Q3 labels (H / E / C)

4. Compute Q3 distribution
   - Count residues in each Q3 class

5. Find longest runs
   - Iterate over the Q3 label list and track the current run length

6. Build the feature matrix
   - Convert residue names to one-letter codes
   - Implement one_hot() and sliding_window_encode() from the Week 2 notes
   - Separate rows by Q3 label; average the 340-dimensional vectors per class

7. Write up your report
```

### Code snippets

**Installation:**

    pip install biopython numpy pydssp

**Downloading and parsing an mmCIF file:**

    from Bio.PDB import PDBList, MMCIFParser

    pdbl = PDBList()
    filename = pdbl.retrieve_pdb_file("1UBQ", file_format="mmCif", pdir=".")   # saves as 1ubq.cif

    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure("protein", filename)
    chain = structure[0]["A"]   # model 0, chain A

    # Print first 5 CA atoms
    count = 0
    for residue in chain:
        for atom in residue:
            if atom.get_name() == "CA" and count < 5:
                print(residue.resname, residue.id[1], atom.get_coord())
                count += 1

**Assigning secondary structure with pydssp:**

    import numpy as np
    import pydssp

    BACKBONE = ["N", "CA", "C", "O"]
    coords = []
    residue_info = []   # (resname, res_seq_num)

    for residue in chain:
        if residue.id[0] != " ":   # skip HETATM records
            continue
        try:
            row = [residue[atom].get_coord() for atom in BACKBONE]
            coords.append(row)
            residue_info.append((residue.resname, residue.id[1]))
        except KeyError:
            pass   # skip residues missing backbone atoms

    coords = np.array(coords, dtype=np.float32)   # shape (L, 4, 3)
    ss3 = pydssp.assign(coords, out_type="c3")    # array of '-', 'H', 'E'

    Q3_MAP = {"-": "C", "H": "H", "E": "E"}
    q3_labels = [Q3_MAP[s] for s in ss3]

    for label in ("H", "E", "C"):
        n = q3_labels.count(label)
        print(f"{label}: {n} ({100 * n / len(q3_labels):.1f}%)")

    print("".join(q3_labels))   # full Q3 string

**Sliding window one-hot encoding:**

    AA_ORDER = "ACDEFGHIKLMNPQRSTVWY"
    aa_to_idx = {aa: i for i, aa in enumerate(AA_ORDER)}

    THREE_TO_ONE = {
        "ALA":"A","CYS":"C","ASP":"D","GLU":"E","PHE":"F",
        "GLY":"G","HIS":"H","ILE":"I","LYS":"K","LEU":"L",
        "MET":"M","ASN":"N","PRO":"P","GLN":"Q","ARG":"R",
        "SER":"S","THR":"T","VAL":"V","TRP":"W","TYR":"Y",
    }

    sequence = "".join(THREE_TO_ONE.get(r, "X") for r, _ in residue_info)

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
            w = padded[i : i + window]
            features.append(np.concatenate([one_hot(aa) for aa in w]))
        return np.array(features)   # shape: (L, window * 20)

    X = sliding_window_encode(sequence, window=17)
    print(X.shape)   # (L, 340)

→ [Biopython PDB docs](https://biopython.org/docs/latest/Tutorial/chapter_pdb.html)  

---

## Optional

Generate per-residue ESM2 embeddings for your protein and visualise them. 

*This wont give any bonus points its just for your understanding*

- Use `esm2_t6_8M_UR50D` (the smallest ESM2 model — fast to run on CPU).
- Extract the representation from the last layer for each residue.
- Reduce to 2D with PCA (`sklearn.decomposition.PCA`) and make a scatter plot coloured by Q3 label (H = red, E = blue, C = grey).
- Write 2–3 sentences: do the embedding clusters correspond to secondary structure classes? What does this tell you about what the language model has learned?

**Starter code:**

    # pip install fair-esm torch scikit-learn matplotlib
    import esm, torch

    model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
    batch_converter = alphabet.get_batch_converter()
    model.eval()

    data = [("protein", "MNIFEMLRID")]   # replace with your sequence
    _, _, tokens = batch_converter(data)

    with torch.no_grad():
        results = model(tokens, repr_layers=[6])
    embeddings = results["representations"][6][0, 1:-1, :].numpy()
    # Shape: (L, 320)  — rows correspond to residues 0..L-1

→ [ESM2 GitHub](https://github.com/facebookresearch/esm)

---

## Submission Format

Submit a single PDF or Markdown file (with any code in a `.py` file or notebook attached) containing:

- [ ] PDB ID, protein name, organism, method, resolution, and chosen chain ID
- [ ] Table of first 5 Cα ATOM records (residue name, number, X/Y/Z)
- [ ] Q3 distribution table (H / E / C counts and percentages)
- [ ] Longest continuous helix and strand with residue positions
- [ ] Per-residue Q3 label string (printed in full)
- [ ] Feature matrix shape
- [ ] Top-3 enriched amino acids in H-windows and E-windows, with 2–4 sentence interpretation
- [ ] Code used (`.py` file, notebook, or inline code block)

**Deadline**: Jun 14, 2026 — results posted to the [leaderboard](../../leaderboard.md) within 48 hours.

---

## Marking Criteria

| Component | Weight | What we look for |
|-----------|--------|-----------------|
| **Correctness** | 60% | Accurate Q3 distributions, correctly computed feature matrix, factually sound amino acid propensity interpretation |
| **Efficiency** | 20% | Clean, non-repetitive code; sensible use of NumPy vectorisation; no unnecessary steps |
| **Clarity and Explanation** | 20% | Well-structured report, complete tables, clear interpretation of amino acid enrichment results |

### Correctness breakdown (60 pts)

| Sub-component | Points |
|---------------|--------|
| Structure metadata and Cα table correctly reported | 10 |
| Q3 distribution accurate | 15 |
| Longest helix and strand positions accurate | 10 |
| Feature matrix shape correct and encoding logic sound | 10 |
| Amino acid enrichment computed correctly and interpreted accurately | 15 |

---

[← Week 2 Topics](index.md) | [Leaderboard →](../../leaderboard.md)
