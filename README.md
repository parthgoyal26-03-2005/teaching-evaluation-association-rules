# Association rules on synthetic teaching evaluations

Academic simulation comparing Apriori, FP-Growth and a vertical-TID ECLAT implementation on 200 generated responses. The data is fictional; it does not measure actual teaching quality or involve real participants. The generator uses a fixed seed. Review `data/DATA_DICTIONARY.md` for the assumptions.

## Method and results

Minimum support is 0.15, maximum itemset length 3, minimum confidence 0.60, and lift must exceed 1. Each algorithm found 55 itemsets and the shared rule generator found 19 qualifying rules. The script validates the itemsets against exhaustive enumeration and checks rule metrics directly. Recorded timings in `results/` are from a small synthetic dataset and cannot establish general algorithm superiority. The 20 timing trials per algorithm vary with the machine and software.

## Reproduce

Use Python 3.12. From this directory:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python scripts/generate_data.py
python scripts/compare_algorithms.py
```

Re-running overwrites the output files in `results/` and changes timing measurements. The CSV dataset is reproducibly generated; timings are not. `results/association_rules.csv` contains the full rule set. These descriptive associations do not establish causation or generalize to actual students.
