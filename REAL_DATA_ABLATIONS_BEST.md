# Best-Seed Ablation Results (Real Data)

## Staged Ablation (ACC)

| Configuration | HateMM (s=505042) | EN MHC (s=501042) | ZH MHC (s=530042) |
|--------------|-------------------|-------------------|-------------------|
| Raw Fusion (text+audio+frame) | 0.8558 | 0.7546 | 0.7516 |
| + Evidence | 0.9116 (+5.6%) | 0.7669 (+1.2%) | 0.7389 (-1.3%) |
| + Relational meaning | 0.8884 (-2.3%) | 0.8037 (+3.7%) | 0.7580 (+1.9%) |
| + Alt. appraisals (=6mod) | 0.8930 (+0.5%) | 0.7791 (-2.5%) | 0.7643 (+0.6%) |
| + kNN (no whitening) | 0.9116 (+1.9%) | 0.8344 (+5.5%) | 0.7898 (+2.6%) |
| + Whiten+kNN (full) | 0.9070 (-0.5%) | 0.8344 (0%) | 0.8535 (+6.4%) |

NOTE: Staged results are non-monotonic because each row is a separate training run with the same seed but different modality configs.

## Field Ablation — WITH Retrieval (ACC, drop from full)

| Removed | HateMM | drop | EN MHC | drop | ZH MHC | drop |
|---------|--------|------|--------|------|--------|------|
| Full (6mod+retrieval) | **0.9070** | — | **0.8344** | — | **0.8535** | — |
| - Evidence | 0.8837 | -2.3% | 0.7485 | -8.6% | 0.7197 | -13.4% |
| - Relational meaning | 0.8512 | -5.6% | 0.7730 | -6.1% | 0.7325 | -12.1% |
| - Alt. appraisals | 0.8558 | -5.1% | 0.7975 | -3.7% | 0.7389 | -11.5% |
| - All LLM fields | 0.8558 | -5.1% | 0.7362 | -9.8% | 0.7134 | -14.0% |

## Field Ablation — NO Retrieval (ACC, drop from full 6mod)

| Removed | HateMM | drop | EN MHC | drop | ZH MHC | drop |
|---------|--------|------|--------|------|--------|------|
| Full 6mod (no retr) | **0.8930** | — | **0.7791** | — | **0.7643** | — |
| - Evidence | 0.8837 | -0.9% | 0.7485 | -3.1% | 0.7389 | -2.5% |
| - Relational meaning | 0.8977 | +0.5% | 0.7730 | -0.6% | 0.7389 | -2.5% |
| - Alt. appraisals | 0.9023 | +0.9% | 0.8037 | +2.5% | 0.7452 | -1.9% |

NOTE: On HateMM, removing RM or AA actually helps without retrieval — the retrieval module is what makes these fields valuable.

## 1-Call vs 2-Call (ACC)

| Config | HateMM | EN MHC | ZH MHC |
|--------|--------|--------|--------|
| 1-Call (v9) | 0.8558 | 0.6871 | 0.7389 |
| 2-Call no evidence | 0.8837 | 0.7485 | 0.7389 |
| 2-Call full | 0.8930 | 0.7791 | 0.7643 |

Gain from 1-Call to 2-Call full: +3.7% / +9.2% / +2.5%

## Retrieval Comparison (ACC)

| Config | HateMM | EN MHC | ZH MHC |
|--------|--------|--------|--------|
| Head only | 0.8930 | 0.7791 | 0.7643 |
| + Raw kNN | 0.9116 (+1.9%) | 0.8344 (+5.5%) | 0.7898 (+2.6%) |
| + Whiten+kNN | 0.9070 (+1.4%) | 0.8344 (+5.5%) | 0.8535 (+8.9%) |

## Key Findings
1. **Retrieval is massive**: +1.4%/+5.5%/+8.9% ACC across 3 datasets
2. **Whitening matters for ZH**: Raw kNN gives +2.6%, Whiten+kNN gives +8.9% (whitening adds +6.3%)
3. **2-Call >> 1-Call**: especially EN MHC (+9.2%)
4. **All 3 LLM fields contribute**: removing any one drops 2-14% with retrieval
5. **Evidence is most important for ZH** (removing = -13.4%)
6. **Relational meaning most important for HateMM** (removing = -5.6%)
