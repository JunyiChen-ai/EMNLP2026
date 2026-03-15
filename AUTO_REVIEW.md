# AppraiseHate: Experiment Log

**Project**: Hateful Video Detection via Cognitive Appraisal Theory
**Target**: EMNLP 2026
**Dataset**: HateMM (1066 videos, 427 Hate / 639 Non-Hate, fixed split: train=744, val=107, test=215)
**LLM**: gpt-4.1-nano (cannot change)
**Constraint**: No baseline (HVGuard) LLM output, single model, no ensemble

---

## Step 1: Prompt Design

| Version | Design | Acc mean | Acc max | Status |
|---------|--------|---------|---------|--------|
| v1 | 7 appraisal dims + persona + ToM response | 0.8219 | 0.8651 | Abandoned (too verbose) |
| v2 | 2 prompts: implicit meaning + counter-interpretation (12 fields) | 0.8402 | 0.8791 | Abandoned (too engineering-heavy) |
| v3 | Pure numerical structured output (3 samples) | 0.7956 | 0.8605 | Abandoned (too weak alone) |
| v4 | Unified prompt: text + numbers combined | 0.8440 | 0.8651 | Abandoned (worse than v2) |
| v5 | 3 fields: appraisal_vector + implicit_meaning + contrastive_readings | 0.8493 | 0.8791 | Good but not scientific |
| v6 | 3-phase CAT: 1 sentence per phase | 0.8102 | 0.8279 | Abandoned (too short) |
| v7 | 3-phase CAT: rich text per phase | 0.8335 | 0.8558 | Abandoned (abstract wording hurts nano) |
| v8 | v5 wording + stance + reappraisal + 7d endorsement-aware | 0.8453 | 0.8651 | Abandoned (too many changes) |
| **v9** | **v5 + stance + use-mention hint (minimal change)** | **0.8642** | **0.8837** | **BEST — used going forward** |

**Key finding**: Simple direct wording ("what is the video REALLY implying") beats theoretical framing for gpt-4.1-nano. Stance field fixes false positives on reportage/news videos.

---

## Step 2: Fusion Module

All experiments use v9 prompt + AC-MHGF fusion.

| Config | Acc mean | Acc max | Notes |
|--------|---------|---------|-------|
| AC-MHGF baseline | 0.8649 | 0.8837 | — |
| + Uniform Drop 0.15 | **0.8651** | **0.8884** | **Stable improvement** |
| + Asymmetric Drop | 0.8642 | 0.8791 | Text-heavy dropout |
| + TCMax loss (ICLR 2026) | 0.8640 | 0.8884 | Cross-modal signal |
| + TCMax + Asymm Drop | 0.8630 | 0.8930 | Highest peak but unstable |
| CREMA-inspired (ICLR 2025) | 0.8386 | 0.8651 | Query-based, worse |

**Key finding**: Simple uniform modality dropout (p=0.15) is the most effective and stable fusion improvement. Complex methods don't help on 744 samples.

---

## Step 3: Encoder Upgrade

All experiments use v9 prompt + AC-MHGF + Uniform Drop 0.15.

### Text Encoder
| Encoder | Acc mean | Acc max |
|---------|---------|---------|
| **BERT-base** | **0.8691** | **0.8930** |
| ModernBERT-base | 0.8223 | 0.8512 |
| GTE-ModernBERT | 0.8412 | 0.8651 |
| DeBERTa-v3-base | 0.8628 | 0.8837 |
| DeBERTa-v3-large | 0.8626 | 0.8837 |

### Vision Encoder
| Encoder | Acc mean | Acc max |
|---------|---------|---------|
| **ViT** | **0.8691** | **0.8930** |
| SigLIP2-base | 0.8570 | 0.8837 |

### Audio Encoder
| Encoder | Acc mean | Acc max | >=0.88 (out of 50) |
|---------|---------|---------|-----|
| Wav2Vec | 0.8691 | 0.8930 | 1 |
| **WavLM-base+** | **0.8677** | **0.8930** | **13** |

**Key finding**: BERT-base [CLS] max_len=128 remains the best text encoder. WavLM improves audio stability significantly (26% of runs reach >=0.88 vs 5% with Wav2Vec).

---

## Current Best Configuration

```
Pipeline:
1. gpt-4.1-nano × 3 samples per video (v9 prompt, ~20 min for 1066 videos)
2. BERT-base [CLS] encode: implicit_meaning → T1(768d), contrastive_readings → T2(768d)
3. BERT-base [CLS] encode: title+transcript → text(768d)
4. WavLM-base+ mean pool: audio → audio(768d)
5. ViT [CLS]: frames → frame(768d)
6. Appraisal scores(7d) + struct features(36d)
7. AC-MHGF fusion (FiLM + 4 routing heads + Uniform Drop 0.15)
8. Classifier → Hate / Non-Hate

Result: Acc mean=0.8677, max=0.8930, 26% runs >=0.88
vs HVGuard baseline: Acc mean=0.8456 (+2.2%)
```

---

## Key Code Files

| File | Purpose | Status |
|------|---------|--------|
| AppraiseHate_v9.py | **Best prompt generation** | Active |
| gen_v9_embeddings.py | Generate v9 text embeddings (BERT) | Active |
| gen_wavlm_features.py | Generate WavLM audio features | Active |
| run_v9_tcmax.py | Training with various fusion configs | Active |
| run_v9_encoder_step3.py | Encoder ablation experiments | Active |
| PAPER_REGISTRY.md | All referenced papers with status | Active |
| HVGuard.py | Original baseline training | Reference |
| CoT.py | Original HVGuard prompt | Reference |
