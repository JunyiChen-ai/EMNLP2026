# Real Ablation Results (5 seeds: 42, 1042, 2042, 3042, 4042)

## Staged Ablation (mean ACC)

| Configuration | HateMM | EN MHC | ZH MHC |
|--------------|--------|--------|--------|
| Raw Fusion (text+audio+frame) | 0.8605 | 0.7227 | 0.7516 |
| + Evidence ledger | 0.8698 (+0.9%) | 0.7485 (+2.6%) | 0.7567 (+0.5%) |
| + Relational meaning | 0.8837 (+1.4%) | 0.7669 (+1.8%) | 0.7554 (-0.1%) |
| + Alt. appraisals (=6mod) | 0.8893 (+0.6%) | 0.7840 (+1.7%) | 0.7694 (+1.4%) |
| + kNN (no whitening) | 0.8874 (-0.2%) | 0.7742 (-1.0%) | 0.7720 (+0.3%) |
| + Whiten+kNN (full) | 0.8912 (+0.4%) | 0.7742 (0%) | 0.7287 (-4.3%) ← PROBLEM |

## Field Ablation (no retrieval, mean ACC)

| Removed Field | HateMM | EN MHC | ZH MHC |
|--------------|--------|--------|--------|
| Full 6mod | 0.8893 | 0.7840 | 0.7694 |
| - Evidence | 0.8726 (-1.7%) | 0.7534 (-3.1%) | 0.7783 (+0.9%) ← weird |
| - Relational meaning | 0.8735 (-1.6%) | 0.7656 (-1.8%) | 0.7656 (-0.4%) |
| - Alt. appraisals | 0.8707 (-1.9%) | 0.7681 (-1.6%) | 0.7694 (0%) |
| - All LLM fields | 0.8605 (-2.9%) | 0.7227 (-6.1%) | 0.7516 (-1.8%) |

## 1-Call vs 2-Call (mean ACC)

| Config | HateMM | EN MHC | ZH MHC |
|--------|--------|--------|--------|
| 1-Call (v9) | 0.8865 | 0.7276 | 0.7414 |
| 2-Call no evidence | 0.8726 | 0.7534 | 0.7783 |
| 2-Call full (t1+t2+ev) | 0.8893 | 0.7840 | 0.7694 |

## Calibration/Retrieval (mean ACC)

| Config | HateMM | EN MHC | ZH MHC |
|--------|--------|--------|--------|
| Head only | 0.8893 | 0.7840 | 0.7694 |
| + Raw kNN | 0.8865 | 0.7706 | 0.7732 |
| + Whiten+kNN | 0.8912 | 0.7742 | 0.7287 ← PROBLEM |

## Issues Found
1. ZH MHC: Whiten+kNN HURTS (-4.3% ACC) — whitening destroys ZH performance
2. ZH MHC: Removing evidence HELPS (+0.9%) — evidence is noise for ZH
3. EN MHC: kNN doesn't help on mean (only helps on specific seeds/max)
4. HateMM: 1-Call v9 (0.8865) ≈ 2-Call full (0.8893) — marginal difference
5. ZH MHC: 2-Call no evidence (0.7783) > 2-Call full (0.7694) — evidence hurts ZH
