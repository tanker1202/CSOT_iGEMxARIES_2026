# Week 1 — Sequences, Databases & Alignment

[← Home](../README.md) | [Week 1 Assignment →](week1-assignment.md)

---

> **No biology background? Perfect.** This page assumes you are starting from zero. Every concept is built up from scratch before we get to the computational tools.

---

## Topics

1. [The Central Dogma — How Life Stores and Uses Information](#1-the-central-dogma)
2. [Amino Acids — The Building Blocks of Proteins](#2-amino-acids--the-building-blocks-of-proteins)
3. [Finding and Fetching Sequences — FASTA, NCBI, UniProt, Biopython](#3-finding-and-fetching-sequences)
4. [Sequence Alignment — Comparing Proteins Computationally](#4-sequence-alignment)

---

## 1. The Central Dogma

### Start here: what is a cell trying to do?

Every living cell — whether in your body, a bacterium, or a yeast — needs to build **proteins**. Proteins do almost everything in biology: they catalyse chemical reactions (as enzymes), carry oxygen (haemoglobin), fight infection (antibodies), provide structure (collagen), and much more.

The cell's problem is: *how does it know which proteins to build, and in what order?*

The answer is stored in **DNA**.

### DNA: the instruction manual

Think of DNA as a very long instruction manual written in a 4-letter alphabet:

```
A   Adenine
T   Thymine
G   Guanine
C   Cytosine
```

A typical human gene is a few hundred to a few thousand of these letters in a row, e.g.:

```
...ATGGCCTTCGAGAAGACGTTCAACCTG...
```

These letters encode instructions for building one protein. The entire human genome contains about **3 billion** of these letters, encoding roughly **20,000 genes**.

### The Central Dogma: DNA → RNA → Protein

The cell does not build proteins directly from DNA. It uses a two-step process:

```
DNA  ──(Transcription)──►  RNA  ──(Translation)──►  Protein
```

**Step 1 — Transcription**: A molecular machine called RNA Polymerase reads a gene in the DNA and makes a temporary copy of it called **messenger RNA (mRNA)**. RNA uses almost the same 4-letter alphabet as DNA, except T is replaced by U (Uracil).

**Step 2 — Translation**: Another molecular machine called the **ribosome** reads the mRNA and builds a protein by stringing together **amino acids** in the order specified by the code. Every group of 3 RNA letters (called a **codon**) specifies one amino acid.

| DNA codon | RNA codon | Amino acid |
|-----------|-----------|------------|
| ATG | AUG | Methionine (start) |
| TTT | UUU | Phenylalanine |
| GGT | GGU | Glycine |
| TAA | UAA | Stop (end of protein) |

### Why does this matter for bioinformatics?

If you know the DNA sequence of a gene, you can **predict** the amino acid sequence of the protein it encodes. Conversely, if you find an unknown DNA sequence, you can search databases of known proteins to figure out what it might do. Almost everything in this course flows from this idea.

---

## 2. Amino Acids — The Building Blocks of Proteins

### What is an amino acid?

Amino acids are small molecules. When many of them are joined end-to-end, they form a **protein chain** (also called a polypeptide). There are exactly **20 standard amino acids** used in life on Earth.

Think of the 20 amino acids as a 20-letter alphabet. A protein is a "word" written in that alphabet — for example, human insulin is 51 amino acids long, while the muscle protein titin is about 34,000 amino acids long.

Each amino acid has:
- A **one-letter code** (used in databases and alignment tools)
- A **three-letter code** (used in scientific papers)
- A **side chain** — a chemical group that gives it a distinct personality

### The 20 amino acids and their one-letter codes

| One-letter | Three-letter | Name |
|------------|--------------|------|
| A | Ala | Alanine |
| C | Cys | Cysteine |
| D | Asp | Aspartate |
| E | Glu | Glutamate |
| F | Phe | Phenylalanine |
| G | Gly | Glycine |
| H | His | Histidine |
| I | Ile | Isoleucine |
| K | Lys | Lysine |
| L | Leu | Leucine |
| M | Met | Methionine |
| N | Asn | Asparagine |
| P | Pro | Proline |
| Q | Gln | Glutamine |
| R | Arg | Arginine |
| S | Ser | Serine |
| T | Thr | Threonine |
| V | Val | Valine |
| W | Trp | Tryptophan |
| Y | Tyr | Tyrosine |

So a short protein might look like this in one-letter code: `MNIFEMLRID` — that is a real 10-amino-acid sequence.

### Grouping amino acids by their chemistry

Not all amino acids are alike. Their side chains have different chemical properties, and this shapes how proteins fold and function:

| Group | What it means | Amino acids |
|-------|--------------|-------------|
| **Nonpolar / Hydrophobic** | Repels water; tends to hide inside the protein | G, A, V, L, I, P, F, M, W |
| **Polar, Uncharged** | Can form hydrogen bonds; often found on the surface | S, T, C, Y, N, Q |
| **Positively Charged (Basic)** | Carries a + charge at neutral pH | K, R, H |
| **Negatively Charged (Acidic)** | Carries a − charge at neutral pH | D, E |

### Why does the chemistry matter?

When comparing two proteins, we care not just about *identical* positions but also *similar* ones. For example, if one protein has a Leucine (L) at position 47 and a related protein has a Valine (V) at the same position, both are hydrophobic — the change is **conservative** and the protein likely still works the same way.

This concept of chemical similarity is central to understanding alignment scores in Section 4.

---

## 3. Finding and Fetching Sequences

### Proteins as text strings

Once you accept that a protein is just a sequence of amino acids, you can represent it as a string of letters:

```
IVGGYTCGANTVPYQVSLNSGYHFCGGSLINSQWVVSAAHCYK
```

That is a real fragment of bovine trypsin. Bioinformatics is, in large part, the science of analysing strings like this — comparing them, searching for patterns, and drawing biological conclusions.

### Where are sequences stored?

Scientists have been sequencing genes and proteins for decades. All of this data is stored in **public databases** that anyone can access for free.

#### NCBI

The **National Center for Biotechnology Information** (ncbi.nlm.nih.gov) is the largest public repository for biological data. The databases most relevant to this course:

| Database | What it stores |
|----------|---------------|
| **GenBank** | Raw DNA/RNA sequences submitted by researchers |
| **RefSeq** | Curated, non-redundant reference sequences |
| **Protein** | Protein sequences from across all databases |
| **BLAST** | A search tool: "find me sequences similar to this one" |

Every sequence in NCBI has an **accession number** — a unique ID like `NP_001077415.1`.

#### UniProt

**UniProt** (uniprot.org) is the gold standard for protein sequences and annotations. It has two tiers:

| Tier | Name | What it means |
|------|------|--------------|
| Reviewed | **Swiss-Prot** | A human expert has read the literature and manually annotated this entry — very trustworthy |
| Unreviewed | **TrEMBL** | Automatically imported and annotated by software — useful but less verified |

A UniProt accession looks like `P00760`. Open any UniProt entry and you will find not just the sequence but also: the protein's function, which residues are in the active site, known disease mutations, 3D structure links, and taxonomy.

**Always prefer Swiss-Prot entries when they exist.**

### The FASTA format

When you download a sequence from NCBI or UniProt, it comes as a **FASTA file** — a simple plain-text format:

```
>sp|P00760|TRY1_BOVIN Cationic trypsin OS=Bos taurus OX=9913
MNNLFQLGVAVAALAALGLAARSVPAGEQARLSQTSGEGKIVGGYTCGANTVPYQVSLNS
GYHFCGGSLINSQWVVSAAHCYKSRIQVRLGEHNIEVLEGNEQFINAAKIITHPNFNGNT
LDNDIMLIKLSSPAVINSRVVHPIVQKLPKEMKPNASSNFTCGGSIGSTGSSSGSAPNQL
```

- The `>` line is the **header**: it contains the accession number, protein name, and organism.
- Everything after it is the sequence, wrapped at ~60 characters per line for readability.
- A single `.fasta` file can contain multiple sequences, each starting with its own `>` header.

### Fetching sequences with Biopython

**Biopython** is a Python library that wraps common bioinformatics tasks so you do not have to write everything from scratch. Install it once:

```
pip install biopython
```

**Reading a local FASTA file:**

    from Bio import SeqIO

    for record in SeqIO.parse("my_sequences.fasta", "fasta"):
        print(record.id)        # accession / name
        print(len(record.seq))  # length in amino acids
        print(record.seq[:10])  # first 10 residues

**Fetching a sequence directly from NCBI:**

    from Bio import Entrez, SeqIO

    Entrez.email = "your@email.com"   # NCBI requires an email address
    handle = Entrez.efetch(
        db="protein",
        id="P00760",
        rettype="fasta",
        retmode="text"
    )
    record = SeqIO.read(handle, "fasta")
    print(record.seq)

---

## 4. Sequence Alignment

### The core idea

Imagine two related proteins — say, the enzyme trypsin from a human and from a cow. They do the same job (cut other proteins at specific sites) and they evolved from a common ancestor millions of years ago. Their sequences are similar but not identical.

**Sequence alignment** is the process of lining up the two sequences so that positions that descended from the same ancestral position are placed in the same column:

```
Human trypsin:  IVGGYTCAANSVPYQVSLNS-GSHFCGGSLINSQWVVS
Bovine trypsin: IVGGYTCGANTVPYQVSLNS-GYHFCGGSLINSQWVVS
                *******.** ********* * *****************
```

Matching columns reveal **conservation** — positions that have not changed are likely critical for function.

### Global vs local alignment

| Type | What it does | Algorithm | When to use |
|------|-------------|-----------|-------------|
| **Global** | Aligns the sequences end-to-end across their full length | Needleman-Wunsch (1970) | Proteins of similar length that are homologous throughout |
| **Local** | Finds the best-matching sub-region between the two sequences | Smith-Waterman | When only a domain or motif is shared |

For this week's assignment, a **global alignment** of two full-length homologous proteins is appropriate.

### Gap penalties

Real related sequences are not just mismatches — evolution also introduces **insertions and deletions** (indels), which appear as gaps (`-`) in the alignment. The alignment algorithm penalises gaps to avoid placing them everywhere:

- **Gap open penalty**: the cost of starting a new gap (higher).
- **Gap extension penalty**: the cost of making an existing gap one residue longer (lower, because one long gap is more likely than many short ones).

### Substitution matrices: not all mismatches are equal

If position 47 in protein A is Leucine (L) and in protein B it is Valine (V), that is a very different situation from A having Leucine and B having Lysine (K) — the first swap is between two hydrophobic residues, the second is between a hydrophobic and a charged residue.

A **substitution matrix** captures this by assigning a score to every possible amino acid pair. The most commonly used matrix is **BLOSUM62**:

- Positive score → the pair appears more often than chance in related proteins (conservative substitution).
- Negative score → the pair rarely appears at the same position in related proteins (radical substitution).
- Highest scores → identical amino acids (e.g., L vs L scores +4).

You do not need to memorise the matrix — alignment tools use it automatically. But understanding *why* it exists helps you interpret % similarity correctly.

### Reading alignment output: % identity and % similarity

After running an alignment you will see two key numbers:

| Metric | How it is calculated | What it tells you |
|--------|---------------------|------------------|
| **% Identity** | (identical positions) / (aligned length) × 100 | How many positions are exactly the same |
| **% Similarity** | (identical + conserved substitutions) / (aligned length) × 100 | How many positions are the same or chemically equivalent |

A pair of proteins with >40% identity almost certainly shares the same fold and function. Below ~20% identity, the relationship is harder to determine from sequence alone (the "twilight zone").

### CLUSTAL Omega

**Clustal Omega** is one of the most widely used tools for sequence alignment — it handles both pairwise and multiple sequence alignments (aligning 3 or more sequences at once).

**Online (no installation needed):** https://www.ebi.ac.uk/Tools/msa/clustalo/

1. Go to the link above.
2. Paste your sequences in FASTA format.
3. Click **Submit**.
4. Download the `.aln` output file.

**Reading the `.aln` output:**

    CLUSTAL O(1.2.4) multiple sequence alignment

    Trypsin_Human    IVGGYTCAANSVPYQVSLNS-GSHFCGGSLINSQWVVS
    Trypsin_Bovine   IVGGYTCGANTVPYQVSLNS-GYHFCGGSLINSQWVVS
                     *******.** ********* * *****************

| Symbol | Meaning |
|--------|---------|
| `*` | Identical residue — fully conserved |
| `:` | Conserved substitution — chemically similar |
| `.` | Semi-conserved substitution — weakly similar |
| (space) | No conservation |

Blocks of `*` symbols indicate **conserved regions** — these are the hotspots you will annotate in this week's assignment.

---

## Resources

- [Biopython Tutorial and Cookbook](https://biopython.org/docs/latest/Tutorial/index.html) — official documentation covering SeqIO, Entrez, pairwise alignment, and more
- [Sequence Alignment — YouTube](https://www.youtube.com/watch?v=u3UiuG2fqhc) — video walkthrough of sequence alignment concepts
- [Needleman & Wunsch (1970)](https://www.sciencedirect.com/science/article/abs/pii/0022283670900574?via%3Dihub) — original paper introducing global sequence alignment

---

[← Home](../README.md) | [Week 1 Assignment →](week1-assignment.md)
