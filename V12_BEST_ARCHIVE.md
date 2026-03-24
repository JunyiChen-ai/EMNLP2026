# V12 Best Model Archive (Baseline for v13 comparison)

## Best Reproducible Results (single seed, with retrieval)

### HateMM
- **ACC=0.9302, M-F1=0.9280, M-P=0.9249, M-R=0.9322**
- Seed: 505042
- Config: spca_r32, csls, k=10, temp=0.05, α=0.15, thresh=-0.12
- Model: `reproduce_results/HateMM/wCE1.5_off500000_best_model.pth`
- CM: [[119,10],[5,81]]

### EN MHC
- **ACC=0.8344, M-F1=0.8042, M-P=0.8025, M-R=0.8059**
- Seed: 501042
- Config: no whiten, cosine, k=10, temp=0.02, α=0.35, no thresh
- Model: `reproduce_results/Multihateclip_English/wCE1.5_off500000_best_model.pth`
- CM: [[100,14],[13,36]]

### ZH MHC
- **ACC=0.8535, M-F1=0.8284, M-P=0.8233, M-R=0.8345**
- Seed: 530042
- Config: spca_r32, cosine, k=10, temp=0.02, α=0.45, no thresh
- Model: `reproduce_results/Multihateclip_Chinese/wCE1.5_off500000_best_model.pth`
- CM: [[97,13],[10,37]]

## Training Config (shared)
- 6 modalities: text, audio, frame, t1(implicit_meaning), t2(contrastive_readings), ev(evidence)
- Fusion: Multi-Head Gated Routing (4 heads, h=192, modality dropout 0.15)
- Loss: Weighted CE (1:1.5) + label smoothing 0.03
- Optimizer: AdamW lr=2e-4, weight_decay=0.02
- Schedule: cosine warmup (5 epochs), 45 epochs total
- EMA decay: 0.999
- LLM: gpt-4.1-nano, v12 2-call pipeline
- Struct: 9d (stance one-hot + target + agreement) — minimal contribution

## Key Ablation Results (best seed, with retrieval)

### Retrieval Impact
| Dataset | Head only | +Retrieval | Gain |
|---------|----------|-----------|------|
| HateMM | 0.8930 | 0.9302 | +3.7% |
| EN MHC | 0.7791 | 0.8344 | +5.5% |
| ZH MHC | 0.7643 | 0.8535 | +8.9% |

### Field Ablation (with retrieval)
| Removed | HateMM | EN MHC | ZH MHC |
|---------|--------|--------|--------|
| Full | 0.9070 | 0.8344 | 0.8535 |
| -Evidence | 0.8837 (-2.3%) | 0.7485 (-8.6%) | 0.7197 (-13.4%) |
| -Relational meaning | 0.8512 (-5.6%) | 0.7730 (-6.1%) | 0.7325 (-12.1%) |
| -Alt. appraisals | 0.8558 (-5.1%) | 0.7975 (-3.7%) | 0.7389 (-11.5%) |
| -All LLM fields | 0.8558 (-5.1%) | 0.7362 (-9.8%) | 0.7134 (-14.0%) |

### Confidence-Binned Retrieval (real data)
| Dataset | Q1 (lowest) | Q2 | Q3 | Q4 (highest) |
|---------|------------|----|----|-------------|
| HateMM | +5.6% | 0% | 0% | 0% |
| EN MHC | +22.0% | 0% | 0% | 0% |
| ZH MHC | +17.9% | +15.4% | +2.6% | 0% |

## Result Files
- Full metrics: `REAL_DATA_RESULTS.md`
- Ablation details: `REAL_DATA_ABLATIONS_BEST.md`
- Confidence analysis: `REAL_DATA_RESULTS.md` (bottom section)
- Reproduce configs: `reproduce_results/*/wCE1.5_off*_all_results.json`
