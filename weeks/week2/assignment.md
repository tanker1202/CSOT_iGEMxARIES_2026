# Week 2 Assignment — PDB Parsing, DSSP Labelling & Sliding Window Encoding

[← Week 2 Topics](index.md) | [Leaderboard →](../../leaderboard.md)

---

## Task

Choose a protein from the table below, download its PDB structure, extract per-residue secondary structure labels using DSSP, and build a sliding window one-hot feature matrix. Submit a report covering all three deliverables.

> This assignment is the foundation for everything ahead: the DSSP labels you extract here are the **ground-truth targets** your ML model will learn to predict in Weeks 3 and 4.

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

### 1. PDB Structure Summary

Parse the PDB file using Biopython and report the following for **one chain of your choice**:

- PDB ID, protein name, organism, experimental method, and resolution (find these on the RCSB page or in the PDB header).
- Chain ID you analysed and total number of residues in that chain.
- A table of the first **5 ATOM records** for the Cα atoms (atom name `CA`) in your chosen chain, showing: residue name, residue number, and X/Y/Z coordinates.

> Tip: filter for `atom.get_name() == "CA"` when iterating over atoms.

### 2. DSSP Secondary Structure Labelling

Run DSSP on your structure and report:

- The full 8-class DSSP distribution — a table showing each label (`H`, `G`, `I`, `E`, `B`, `T`, `S`, `C`) and the count and percentage of residues assigned to it.
- The **Q3 distribution** — collapse the 8 labels into H / E / C using the mapping from the Week 2 notes, and report the count and percentage for each Q3 class.
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
1. Download the PDB file
   - Go to https://www.rcsb.org/structure/<PDBID>
   - Click Download > PDB Format  (or use Biopython PDBList, shown below)

2. Parse the structure
   - Use Bio.PDB.PDBParser to load the file
   - Pick a single chain to analyse

3. Run DSSP
   - Option A (no installation): upload the .pdb file to the DSSP web server at
     https://www3.cmbi.umcn.nl/dssp/
     and download the output
   - Option B (Biopython): Bio.PDB.DSSP — see code snippet below
     (requires dssp / mkdssp binary installed separately)

4. Compute Q3 distribution
   - Map the 8 DSSP labels to H / E / C
   - Count residues in each class

5. Find longest runs
   - Iterate over the Q3 string and track the current run length

6. Build the feature matrix
   - Implement one_hot() and sliding_window_encode() from the Week 2 notes
   - Separate rows by Q3 label; average the 340-dimensional vectors per class

7. Write up your report
```

### Biopython code snippets

**Downloading and parsing a PDB file:**

    from Bio.PDB import PDBList, PDBParser

    pdbl = PDBList()
    pdbl.retrieve_pdb_file("1TGT", file_type="pdb", pdir=".")   # saves as pdb1tgt.ent

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", "pdb1tgt.ent")
    chain = structure[0]["A"]   # model 0, chain A

    # Print first 5 CA atoms
    count = 0
    for residue in chain:
        for atom in residue:
            if atom.get_name() == "CA" and count < 5:
                print(residue.resname, residue.id[1], atom.get_coord())
                count += 1

**Running DSSP with Biopython:**

    from Bio.PDB import PDBParser
    from Bio.PDB.DSSP import DSSP

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", "pdb1tgt.ent")
    model = structure[0]

    dssp = DSSP(model, "pdb1tgt.ent")   # requires mkdssp binary

    Q3_MAP = {
        "H": "H", "G": "H", "I": "H",   # all helix types -> H
        "E": "E", "B": "E",               # strand types    -> E
        "T": "C", "S": "C", "C": "C",    # everything else -> C
        "-": "C"
    }

    for (chain_id, res_id), values in dssp.property_dict.items():
        aa      = values[1]   # amino acid one-letter code
        ss8     = values[2]   # 8-class DSSP label
        ss3     = Q3_MAP.get(ss8, "C")
        print(chain_id, res_id[1], aa, ss8, ss3)

**Sliding window one-hot encoding:**

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
            w = padded[i : i + window]
            features.append(np.concatenate([one_hot(aa) for aa in w]))
        return np.array(features)   # shape: (L, window * 20)

    # Example usage
    sequence = "MNIFEMLRID"   # replace with your actual chain sequence
    X = sliding_window_encode(sequence, window=17)
    print(X.shape)   # (10, 340)

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
- [ ] 8-class DSSP distribution table
- [ ] Q3 distribution table
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
| **Correctness** | 60% | Accurate DSSP labels and Q3 distributions, correctly computed feature matrix, factually sound amino acid propensity interpretation |
| **Efficiency** | 20% | Clean, non-repetitive code; sensible use of NumPy vectorisation; no unnecessary steps |
| **Clarity and Explanation** | 20% | Well-structured report, complete tables, clear interpretation of amino acid enrichment results |

### Correctness breakdown (60 pts)

| Sub-component | Points |
|---------------|--------|
| PDB metadata and Cα table correctly reported | 10 |
| 8-class DSSP distribution accurate | 10 |
| Q3 distribution and longest-run positions accurate | 15 |
| Feature matrix shape correct and encoding logic sound | 10 |
| Amino acid enrichment computed correctly and interpreted accurately | 15 |

---

[← Week 2 Topics](index.md) | [Leaderboard →](../../leaderboard.md)
