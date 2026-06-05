# Week 1 Assignment — Protein Sequence Alignment

[← Week 1 Topics](index.md) | [Leaderboard →](../../leaderboard.md)

---

## Task

Align **two homologous proteins** of your choice using a sequence alignment tool, and submit a report covering all three deliverables below.

> A pair of proteins is **homologous** when they share a common evolutionary ancestor — typically evident from significant sequence similarity (>25% identity over a meaningful length).

### Suggested protein pairs (pick one, or choose your own)

| Pair | Protein 1 | Protein 2 | Why interesting |
|------|-----------|-----------|-----------------|
| A | Human Trypsin ([P07477](https://www.uniprot.org/uniprotkb/P07477)) | Bovine Trypsin ([P00760](https://www.uniprot.org/uniprotkb/P00760)) | Classic serine protease, well-studied |
| B | Human Haemoglobin alpha ([P69905](https://www.uniprot.org/uniprotkb/P69905)) | Horse Haemoglobin alpha ([P01958](https://www.uniprot.org/uniprotkb/P01958)) | Oxygen transport, conserved active site |
| C | Human Lysozyme ([P61626](https://www.uniprot.org/uniprotkb/P61626)) | Hen egg-white Lysozyme ([P00698](https://www.uniprot.org/uniprotkb/P00698)) | Antimicrobial, well-annotated |
| D | Human Papain-like cysteine protease ([P49863](https://www.uniprot.org/uniprotkb/P49863)) | Plant Papain ([P00784](https://www.uniprot.org/uniprotkb/P00784)) | Cross-kingdom comparison |

You may choose your own pair — just justify the choice of homologs in your report.

---

## Deliverables

### 1. Percent Identity and Percent Similarity

Report both metrics from your alignment output.

- **% Identity** = (number of identical positions) / (total aligned positions) × 100
- **% Similarity** = (identical + conserved substitutions) / (total aligned positions) × 100

Include the raw alignment output (`.aln` file or screenshot) as an appendix.

### 2. Annotated Conservation Hotspots

Identify and annotate at least **3 conserved regions** or **conserved individual residues** from your alignment. For each:

- Give the position range (e.g., residues 50–65 in Protein 1 numbering).
- Describe the conservation pattern (fully conserved, conserved substitution, etc.).
- Note whether the region corresponds to a known functional element (active site, binding site, disulfide bond, etc.) — cross-reference the [UniProt](https://www.uniprot.org/) annotations for your proteins.

### 3. Biological Significance

Write a short discussion (200–400 words) interpreting your results:

- What does the degree of sequence identity/similarity tell you about how closely related these proteins are?
- Which conserved regions are most likely to be functionally critical, and why?
- Are there regions of low conservation? What might explain them (surface loops, species-specific adaptations, etc.)?

---

## Recommended Workflow

```
1. Retrieve sequences
   - Go to UniProt (https://www.uniprot.org/) or NCBI Protein
   - Search for your proteins by name or accession
   - Download both sequences as a single FASTA file

2. Run alignment
   - Clustal Omega online: https://www.ebi.ac.uk/Tools/msa/clustalo/
   - Paste both sequences, run with default settings
   - Download the .aln output

3. Extract metrics
   - % identity and % similarity are reported at the top of most alignment outputs

4. Annotate hotspots
   - Look up your proteins on UniProt
   - Cross-reference conserved positions with the annotated features in the entry

5. Write up your report
```

### Optional: Biopython for pairwise alignment

    from Bio import SeqIO
    from Bio.Align import PairwiseAligner
    from Bio.Align import substitution_matrices

    seq1 = SeqIO.read("protein1.fasta", "fasta").seq
    seq2 = SeqIO.read("protein2.fasta", "fasta").seq

    aligner = PairwiseAligner()
    aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")
    aligner.open_gap_score = -11
    aligner.extend_gap_score = -1

    alignments = aligner.align(seq1, seq2)
    print(alignments[0])

→ [Biopython pairwise alignment docs](https://biopython.org/docs/latest/Tutorial/chapter_pairwise.html)

---

## Submission Format

Submit a single PDF or Markdown file containing:

- [ ] Protein names, species, and UniProt/NCBI accession numbers
- [ ] % Identity and % Similarity values with the alignment output attached
- [ ] Table or annotated alignment showing at least 3 conserved hotspots
- [ ] Biological significance discussion (200–400 words)
- [ ] Code used (if any) or tool settings

**Deadline**: Jun 7, 2026 — results posted to the [leaderboard](../../leaderboard.md) within 48 hours.

---

## Marking Criteria

| Component | Weight | What we look for |
|-----------|--------|-----------------|
| **Correctness** | 60% | Accurate alignment metrics, correctly identified conserved positions, factually sound biological interpretation |
| **Efficiency** | 20% | Appropriate tool choice, clean workflow, no redundant steps; if code is submitted — readable and non-repetitive |
| **Clarity and Explanation** | 20% | Well-structured report, clear annotation of hotspots, concise and accurate biological discussion |

### Correctness breakdown (60 pts)

| Sub-component | Points |
|---------------|--------|
| Correct % identity and % similarity reported | 15 |
| Hotspots accurately identified from alignment | 20 |
| Hotspots cross-referenced with UniProt annotations | 10 |
| Biological significance discussion is factually accurate | 15 |

---

[← Week 1 Topics](index.md) | [Leaderboard →](../../leaderboard.md)
