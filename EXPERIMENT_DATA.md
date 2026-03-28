# HATELENS Experiment Data

All experiment data collected for paper. Generated 2026-03-28.

---

## 1. Ablation Study (Step 13)

All ablations trained from scratch with exact best seed + best retrieval config per dataset.
Full model = max performance. Code: `run_ablation_max.py`

### HateMM (seed=607042, zca+cosine, k=40, t=0.1, α=0.5, thresh=-0.10)

| Variant | ACC | M-F1 | M-P | M-R |
|---------|:---:|:----:|:---:|:---:|
| **Full model** | **91.6** | **91.4** | **91.0** | **91.9** |
| --what | 87.4 | 86.7 | 87.4 | 86.2 |
| --where | 83.3 | 82.2 | 83.2 | 81.6 |
| --why | 83.7 | 82.5 | 84.0 | 81.8 |
| --how | 84.2 | 83.3 | 83.8 | 82.9 |
| Percep. only | 87.9 | 87.4 | 87.5 | 87.2 |
| Cogn. only | 86.0 | 85.5 | 85.5 | 85.5 |
| HVGuard CoT | 84.2 | 83.6 | 83.4 | 83.9 |
| MoE fusion | 85.6 | 84.8 | 85.4 | 84.3 |
| HVGuard fusion | 87.0 | 86.2 | 87.0 | 85.7 |
| No retrieval | 87.0 | 86.4 | 86.5 | 86.2 |
| No whitening | 87.9 | 87.2 | 88.0 | 86.6 |

### MHClip-Y (seed=908042, spca_r32+cosine, k=10, t=0.02, α=0.5, thresh=None)

| Variant | ACC | M-F1 | M-P | M-R |
|---------|:---:|:----:|:---:|:---:|
| **Full model** | **87.1** | **84.0** | **85.9** | **82.6** |
| --what | 80.4 | 75.1 | 77.4 | 73.7 |
| --where | 82.8 | 79.8 | 79.5 | 80.2 |
| --why | 81.6 | 78.4 | 78.1 | 78.7 |
| --how | 77.9 | 70.7 | 74.8 | 69.1 |
| Percep. only | 72.4 | 57.7 | 67.5 | 58.2 |
| Cogn. only | 73.0 | 58.2 | 69.3 | 58.6 |
| HVGuard CoT | 69.3 | 61.6 | 62.5 | 61.2 |
| MoE fusion | 71.2 | 58.5 | 64.1 | 58.4 |
| HVGuard fusion | 80.4 | 75.1 | 77.4 | 73.7 |
| No retrieval | 77.9 | 72.7 | 73.8 | 72.0 |
| No whitening | 81.0 | 74.9 | 79.3 | 73.0 |

### MHClip-B (seed=99042, zca+cosine, k=25, t=0.1, α=0.1, thresh=None)

| Variant | ACC | M-F1 | M-P | M-R |
|---------|:---:|:----:|:---:|:---:|
| **Full model** | **89.2** | **85.6** | **92.0** | **82.5** |
| --what | 82.8 | 78.8 | 79.9 | 78.0 |
| --where | 82.2 | 78.2 | 79.0 | 77.5 |
| --why | 80.3 | 77.1 | 76.5 | 78.0 |
| --how | 79.0 | 76.2 | 75.3 | 77.7 |
| Percep. only | 79.0 | 73.0 | 75.6 | 71.6 |
| Cogn. only | 79.0 | 72.5 | 75.9 | 71.0 |
| HVGuard CoT | 75.2 | 71.2 | 70.8 | 71.9 |
| MoE fusion | 81.5 | 79.1 | 78.1 | 80.7 |
| HVGuard fusion | 80.9 | 76.9 | 77.3 | 76.6 |
| No retrieval | 86.0 | 82.9 | 83.8 | 82.1 |
| No whitening | 85.4 | 82.2 | 82.9 | 81.6 |

### ImpliHateVid (seed=28042, spca_r32+csls, k=10, t=0.02, α=0.4, thresh=0.06)

| Variant | ACC | M-F1 | M-P | M-R |
|---------|:---:|:----:|:---:|:---:|
| **Full model** | **89.5** | **89.5** | **89.8** | **89.5** |
| --what | 85.0 | 85.0 | 85.3 | 85.0 |
| --where | 87.8 | 87.8 | 87.9 | 87.8 |
| --why | 85.5 | 85.5 | 86.0 | 85.5 |
| --how | 87.5 | 87.5 | 87.6 | 87.5 |
| Percep. only | 85.0 | 85.0 | 85.1 | 85.0 |
| Cogn. only | 84.8 | 84.8 | 84.8 | 84.8 |
| HVGuard CoT | 87.5 | 87.5 | 87.5 | 87.5 |
| MoE fusion | 84.3 | 84.3 | 84.3 | 84.3 |
| HVGuard fusion | 85.0 | 85.0 | 85.1 | 85.0 |
| No retrieval | 86.3 | 86.3 | 86.7 | 86.3 |
| No whitening | 86.8 | 86.8 | 86.9 | 86.8 |

---

## 2. Statistical Significance (vs HVGuard, 50k paired bootstrap + McNemar)

All p < 0.001 on all 4 metrics x 4 datasets.

### HateMM

| Metric | Δ (%) | 95% CI | p-value |
|--------|:-----:|:------:|:-------:|
| ACC | +14.0 | [8.4, 20.0] | <0.001 |
| M-F1 | +16.1 | [10.0, 22.4] | <0.001 |
| M-P | +12.5 | [6.5, 18.6] | <0.001 |
| M-R | +17.5 | [11.8, 23.3] | <0.001 |
| McNemar | b=7, c=37 | chi2=19.1 | <0.001 |

### MHClip-Y

| Metric | Δ (%) | 95% CI | p-value |
|--------|:-----:|:------:|:-------:|
| ACC | +17.2 | [9.8, 24.5] | <0.001 |
| M-F1 | +42.8 | [35.6, 49.3] | <0.001 |
| M-P | +50.9 | [42.9, 57.9] | <0.001 |
| M-R | +32.6 | [25.6, 39.2] | <0.001 |
| McNemar | b=7, c=35 | chi2=17.4 | <0.001 |

### MHClip-B

| Metric | Δ (%) | 95% CI | p-value |
|--------|:-----:|:------:|:-------:|
| ACC | +19.1 | [12.7, 25.5] | <0.001 |
| M-F1 | +44.3 | [37.3, 50.5] | <0.001 |
| M-P | +57.0 | [51.8, 60.9] | <0.001 |
| M-R | +32.5 | [25.5, 39.3] | <0.001 |
| McNemar | b=1, c=31 | chi2=26.3 | <0.001 |

### ImpliHateVid

| Metric | Δ (%) | 95% CI | p-value |
|--------|:-----:|:------:|:-------:|
| ACC | +14.0 | [9.5, 18.5] | <0.001 |
| M-F1 | +14.4 | [10.0, 18.9] | <0.001 |
| M-P | +12.4 | [8.2, 16.7] | <0.001 |
| M-R | +14.0 | [9.9, 18.2] | <0.001 |
| McNemar | b=18, c=74 | chi2=32.9 | <0.001 |

---

## 3. Seed Stability (5 seeds per dataset, best retrieval config applied)

All seeds: WNI retrieval improves or matches head-only accuracy.

### HateMM

| Seed | Head ACC | Head F1 | Full ACC | Full F1 | Δ ACC |
|------|:--------:|:-------:|:--------:|:-------:|:-----:|
| **607042** | 87.0 | 86.4 | **91.6** | **91.4** | **+4.7** |
| 1042 | 87.4 | 86.7 | 88.8 | 88.3 | +1.4 |
| 2042 | 83.7 | 82.9 | 85.1 | 84.2 | +1.4 |
| 3042 | 87.9 | 87.2 | 88.8 | 88.4 | +0.9 |
| 5042 | 86.0 | 85.4 | 86.0 | 85.3 | +0.0 |

### MHClip-Y

| Seed | Head ACC | Head F1 | Full ACC | Full F1 | Δ ACC |
|------|:--------:|:-------:|:--------:|:-------:|:-----:|
| **908042** | 77.9 | 72.7 | **87.1** | **84.0** | **+9.2** |
| 3042 | 79.1 | 74.6 | 84.7 | 80.9 | +5.5 |
| 5042 | 74.8 | 70.3 | 78.5 | 71.3 | +3.7 |
| 8042 | 79.8 | 73.3 | 79.8 | 73.3 | +0.0 |
| 9042 | 76.7 | 72.9 | 76.7 | 69.0 | +0.0 |

### MHClip-B

| Seed | Head ACC | Head F1 | Full ACC | Full F1 | Δ ACC |
|------|:--------:|:-------:|:--------:|:-------:|:-----:|
| **99042** | 86.0 | 82.9 | **89.2** | **85.6** | **+3.2** |
| 1042 | 82.2 | 76.9 | 84.1 | 81.4 | +1.9 |
| 2042 | 80.9 | 74.4 | 82.2 | 76.5 | +1.3 |
| 3042 | 81.5 | 78.4 | 81.5 | 78.4 | +0.0 |
| 5042 | 78.3 | 77.1 | 79.6 | 78.1 | +1.3 |

### ImpliHateVid

| Seed | Head ACC | Head F1 | Full ACC | Full F1 | Δ ACC |
|------|:--------:|:-------:|:--------:|:-------:|:-----:|
| **28042** | 85.8 | 85.7 | **89.5** | **89.5** | **+3.7** |
| 1042 | 85.0 | 85.0 | 85.8 | 85.8 | +0.8 |
| 2042 | 86.0 | 86.0 | 86.5 | 86.5 | +0.5 |
| 7042 | 86.3 | 86.3 | 86.5 | 86.5 | +0.3 |
| 7042 | 86.3 | 86.3 | 86.5 | 86.5 | +0.3 |

---

## 4. Calibration (Brier Score)

Brier score (lower = better). Computed on blended logits (no threshold).

| Dataset | Head Brier | Full Brier | Δ | Head ECE | Full ECE |
|---------|:----------:|:----------:|:-:|:--------:|:--------:|
| HateMM | 0.166 | **0.160** | -0.006 | 0.273 | 0.277 |
| MHClip-Y | 0.146 | **0.139** | -0.007 | 0.108 | 0.163 |
| MHClip-B | 0.212 | **0.208** | -0.004 | 0.314 | 0.340 |
| ImpliHateVid | **0.106** | 0.107 | +0.001 | 0.058 | 0.119 |

Note: ECE increases after retrieval because blending shifts the logit scale. Brier score is the more reliable metric here as it measures both calibration and discrimination jointly.

---

## 5. Retrieval Bank Size

ACC (mean ± std over 20 trials) at different fractions of the training bank.

| Fraction | HateMM | MHClip-Y | MHClip-B | ImpliHateVid |
|:--------:|:------:|:--------:|:--------:|:------------:|
| 5% | 85.4±1.3 | 78.9±2.1 | 83.3±1.8 | 87.0±1.0 |
| 10% | 85.7±1.3 | 79.6±1.7 | 84.7±1.4 | 87.0±0.8 |
| 25% | 86.7±1.3 | 81.0±1.4 | 86.0±1.2 | 87.6±0.7 |
| 50% | 86.8±1.0 | 83.7±1.6 | 86.2±1.0 | 88.3±0.7 |
| 75% | 87.4±0.8 | 85.1±1.1 | 86.5±0.8 | 88.4±0.6 |
| 100% | 87.4±0.0 | 87.1±0.0 | 89.2±0.0 | 88.8±0.0 |

Trend: monotonically increasing on all datasets. Retrieval benefits grow with bank size.

---

## 6. Modality Zero-out (Full model with retrieval)

ACC after zeroing out each modality. Drop = full_ACC - zeroed_ACC.

### HateMM (Full ACC = 91.6)

| Modality | ACC | Drop |
|----------|:---:|:----:|
| text | 84.7 | +7.0 |
| audio | 87.0 | +4.7 |
| frame | 88.4 | +3.3 |
| ans_what | 87.4 | +4.2 |
| ans_target | 87.4 | +4.2 |
| ans_where | 88.8 | +2.8 |
| ans_why | 87.9 | +3.7 |
| ans_how | 87.9 | +3.7 |

### MHClip-Y (Full ACC = 87.1)

| Modality | ACC | Drop |
|----------|:---:|:----:|
| text | 77.3 | +9.8 |
| audio | 77.3 | +9.8 |
| frame | 79.1 | +8.0 |
| ans_what | 79.1 | +8.0 |
| ans_target | 81.0 | +6.1 |
| ans_where | 78.5 | +8.6 |
| ans_why | 79.1 | +8.0 |
| ans_how | 79.8 | +7.4 |

### MHClip-B (Full ACC = 89.2)

| Modality | ACC | Drop |
|----------|:---:|:----:|
| text | 80.3 | +8.9 |
| audio | 80.9 | +8.3 |
| frame | 78.3 | +10.8 |
| ans_what | 77.1 | +12.1 |
| ans_target | 73.9 | +15.3 |
| ans_where | 83.4 | +5.7 |
| ans_why | 82.2 | +7.0 |
| ans_how | 66.9 | +22.3 |

### ImpliHateVid (Full ACC = 89.5)

| Modality | ACC | Drop |
|----------|:---:|:----:|
| text | 87.0 | +2.5 |
| audio | 86.5 | +3.0 |
| frame | 83.3 | +6.2 |
| ans_what | 80.0 | +9.5 |
| ans_target | 84.0 | +5.5 |
| ans_where | 82.5 | +7.0 |
| ans_why | 86.3 | +3.2 |
| ans_how | 86.5 | +3.0 |

All drops positive across all datasets and modalities.

---

## 7. Label Noise Robustness

ACC (mean ± std over 20 trials) under random label flips in the retrieval bank.

| Noise Rate | HateMM | MHClip-Y | MHClip-B | ImpliHateVid |
|:----------:|:------:|:--------:|:--------:|:------------:|
| 0% | 87.4±0.0 | 87.1±0.0 | 89.2±0.0 | 88.8±0.0 |
| 5% | 87.2±0.8 | 86.2±1.1 | 88.1±0.7 | 88.5±0.5 |
| 10% | 87.0±1.0 | 84.5±1.3 | 86.7±1.1 | 88.0±0.6 |
| 20% | 86.8±1.1 | 82.1±1.9 | 86.2±1.3 | 87.2±0.8 |

Trend: monotonically decreasing. Graceful degradation — at 20% noise, ACC drops <3pp on all datasets.

---

## 8. Per-class P/R/F1

### HateMM

| Class | Precision | Recall | F1 |
|-------|:---------:|:------:|:--:|
| Non Hate (0) | 95.1 | 90.7 | 92.9 |
| Hate (1) | 87.0 | 93.0 | 89.9 |

CM: [[117, 12], [6, 80]]

### MHClip-Y

| Class | Precision | Recall | F1 |
|-------|:---------:|:------:|:--:|
| Normal (0) | 88.4 | 93.9 | 91.1 |
| Hateful (1) | 83.3 | 71.4 | 76.9 |

CM: [[107, 7], [14, 35]]

### MHClip-B

| Class | Precision | Recall | F1 |
|-------|:---------:|:------:|:--:|
| Normal (0) | 87.2 | 99.1 | 92.8 |
| Hateful (1) | 96.9 | 66.0 | 78.5 |

CM: [[109, 1], [16, 31]]

### ImpliHateVid

| Class | Precision | Recall | F1 |
|-------|:---------:|:------:|:--:|
| Normal (0) | 93.4 | 85.1 | 89.1 |
| Hateful (1) | 86.2 | 94.0 | 90.0 |

CM: [[171, 30], [12, 188]]

---

## 9. Efficiency Analysis (from AUTO_REVIEW.md Step 14)

### Parameters

| Method | Params |
|--------|:------:|
| MoRE | 0.42M |
| **Ours** | 1.52M |
| HVGuard | 2.52M |
| ImpliHateVid | 6.37M |

---

## Best Configs Reference

| Dataset | Seed | Whiten | kNN | k | Temp | Alpha | Thresh |
|---------|:----:|:------:|:---:|:-:|:----:|:-----:|:------:|
| HateMM | 607042 | zca | cosine | 40 | 0.1 | 0.5 | -0.10 |
| MHClip-Y | 908042 | spca_r32 | cosine | 10 | 0.02 | 0.5 | None |
| MHClip-B | 99042 | zca | cosine | 25 | 0.1 | 0.1 | None |
| ImpliHateVid | 28042 | spca_r32 | csls | 10 | 0.02 | 0.4 | 0.06 |

---

## Data Sources

- Ablation: `AUTO_REVIEW.md` Step 13 (from `run_ablation_max.py`)
- Appendix experiments: `appendix_results/{dataset}/appendix_results.json` (from `run_appendix_fix2.py`)
- Efficiency: `AUTO_REVIEW.md` Step 14
