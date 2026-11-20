# PII detection

Synthetic corpus: 600 sentences (seed 7, 43% Italian), 909 gold spans (48 in spellings the detectors do not target, such as `name [at] example [dot] com` or dotted phone numbers), 498 hard negatives. Matching: same class and intersection-over-union >= 0.5.

## regex + validators

Overall precision 1.000, recall 0.940, F1 0.969; exact boundaries on 100.0% of matches.

| Class | Precision | Recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|
| CARD | 1.000 | 0.943 | 0.971 | 83 | 0 | 5 |
| CODICE_FISCALE | 1.000 | 1.000 | 1.000 | 82 | 0 | 0 |
| DOB | 1.000 | 0.926 | 0.962 | 100 | 0 | 8 |
| EMAIL | 1.000 | 0.884 | 0.939 | 122 | 0 | 16 |
| IBAN | 1.000 | 0.919 | 0.958 | 68 | 0 | 6 |
| IP | 1.000 | 1.000 | 1.000 | 96 | 0 | 0 |
| PHONE | 1.000 | 0.921 | 0.959 | 151 | 0 | 13 |
| SECRET | 1.000 | 1.000 | 1.000 | 50 | 0 | 0 |

## regex + validators + spaCy names

Overall precision 0.899, recall 0.921, F1 0.910; exact boundaries on 99.3% of matches.

| Class | Precision | Recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|
| CARD | 1.000 | 0.943 | 0.971 | 83 | 0 | 5 |
| CODICE_FISCALE | 1.000 | 1.000 | 1.000 | 82 | 0 | 0 |
| DOB | 1.000 | 0.926 | 0.962 | 100 | 0 | 8 |
| EMAIL | 1.000 | 0.884 | 0.939 | 122 | 0 | 16 |
| IBAN | 1.000 | 0.919 | 0.958 | 68 | 0 | 6 |
| IP | 1.000 | 1.000 | 1.000 | 96 | 0 | 0 |
| PERSON | 0.475 | 0.780 | 0.590 | 85 | 94 | 24 |
| PHONE | 1.000 | 0.921 | 0.959 | 151 | 0 | 13 |
| SECRET | 1.000 | 1.000 | 1.000 | 50 | 0 | 0 |

Reproduce: `python benchmarks/run_benchmarks.py --suite pii`
