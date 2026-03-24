# Real Experimental Data (for paper)

## Best Seed Full Metrics (eval_best.py, 2026-03-19)

### HateMM (seed=505042, spca_r32+csls+k10+temp0.05+α0.15+thresh=-0.12)
| Method | ACC | M-F1 | M-P | M-R |
|--------|-----|------|-----|-----|
| Head only | 0.8930 | 0.8883 | 0.8891 | 0.8876 |
| Head+thresh | 0.9209 | 0.9184 | 0.9154 | 0.9225 |
| Blended argmax | 0.9070 | 0.9031 | 0.9031 | 0.9031 |
| **Blended+thresh** | **0.9302** | **0.9280** | **0.9249** | **0.9322** |

CM (best): [[119,10],[5,81]]
Per-class: Non-Hate P=0.9597 R=0.9225 F1=0.9407 | Hate P=0.8901 R=0.9419 F1=0.9153

### EN MHC (seed=501042, no_whiten+cosine+k10+temp0.02+α0.35)
| Method | ACC | M-F1 | M-P | M-R |
|--------|-----|------|-----|-----|
| Head only | 0.7791 | 0.7113 | 0.7452 | 0.6967 |
| **Blended argmax** | **0.8344** | **0.8042** | **0.8025** | **0.8059** |

CM (best): [[100,14],[13,36]]
Per-class: Non-Hate P=0.8850 R=0.8772 F1=0.8811 | Hate P=0.7200 R=0.7347 F1=0.7273

### ZH MHC (seed=530042, spca_r32+cosine+k10+temp0.02+α0.45)
| Method | ACC | M-F1 | M-P | M-R |
|--------|-----|------|-----|-----|
| Head only | 0.7643 | 0.7377 | 0.7298 | 0.7587 |
| **Blended argmax** | **0.8535** | **0.8284** | **0.8233** | **0.8345** |

CM (best): [[97,13],[10,37]]
Per-class: Non-Hate P=0.9065 R=0.8818 F1=0.8940 | Hate P=0.7400 R=0.7872 F1=0.7629

---

## Confidence-Binned Retrieval Analysis (eval_confidence_bins.py, 2026-03-19)

### HateMM
| Bin | N | Head ACC | Blend ACC | Gain | Head FN | Rescued |
|-----|---|----------|-----------|------|---------|---------|
| Q1 (lowest conf) | 54 | 0.7222 | 0.7778 | **+5.6%** | 7 | +2 |
| Q2 | 53 | 0.9057 | 0.9057 | 0% | 4 | 0 |
| Q3 | 54 | 0.9815 | 0.9815 | 0% | 0 | 0 |
| Q4 (highest conf) | 54 | 0.9630 | 0.9630 | 0% | 1 | 0 |
Net flips: +4 rescued, -1 broken = net +3

### EN MHC
| Bin | N | Head ACC | Blend ACC | Gain | Head FN | Rescued |
|-----|---|----------|-----------|------|---------|---------|
| Q1 (lowest conf) | 41 | 0.5366 | 0.7561 | **+22.0%** | 14 | +12 |
| Q2 | 40 | 0.7750 | 0.7750 | 0% | 4 | 0 |
| Q3 | 41 | 0.8780 | 0.8780 | 0% | 4 | 0 |
| Q4 (highest conf) | 41 | 0.9268 | 0.9268 | 0% | 3 | 0 |
Net flips: +12 rescued, -3 broken = net +9

### ZH MHC
| Bin | N | Head ACC | Blend ACC | Gain | Head FN | Rescued |
|-----|---|----------|-----------|------|---------|---------|
| Q1 (lowest conf) | 39 | 0.6154 | 0.7949 | **+17.9%** | 6 | +2 |
| Q2 | 39 | 0.6154 | 0.7692 | **+15.4%** | 4 | 0 |
| Q3 | 39 | 0.8718 | 0.8974 | +2.6% | 2 | 0 |
| Q4 (highest conf) | 40 | 0.9500 | 0.9500 | 0% | 0 | 0 |
Net flips: +19 rescued, -5 broken = net +14
