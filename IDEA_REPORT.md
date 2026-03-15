# AppraiseHate: Idea & Method Report

**Title**: AppraiseHate: Cognitive Appraisal Theory-Guided Multimodal Reasoning for Hateful Video Detection
**Target**: EMNLP 2026
**Dataset**: HateMM

## Core Idea

Hate is not merely negative sentiment — it is a **structured cognitive appraisal pattern** where a target group is evaluated as blameworthy, threatening, contaminating, inferior, excludable, and/or less-than-human. We use Cognitive Appraisal Theory (CAT) to guide both:
1. **LLM reasoning** (Step 1): structured extraction of appraisal dimensions + implicit meaning + stance
2. **Multimodal fusion** (Step 2): appraisal scores condition how modalities are integrated

## Method

### Step 1: CAT-Informed LLM Reasoning (v9 prompt)
- Single prompt per video, 3 samples for calibration
- 4 output fields: appraisal_vector (7d), implicit_meaning, contrastive_readings, stance
- Key innovation: use-mention distinction ("video can SHOW hate without ENDORSING it")
- Theory framing: "CAT-informed measurement instrument" (Chen & Wang, EMNLP 2025)

### Step 2: Appraisal-Conditioned Multimodal Fusion (AC-MHGF)
- 5 modalities: text + audio + frame + T1(implicit) + T2(contrastive)
- FiLM conditioning from appraisal scores
- 4 score-conditioned routing heads
- Uniform modality dropout (p=0.15)
- Training: AdamW, cosine warmup, EMA, label smoothing

### Encoders
- Text: BERT-base [CLS] (still best for short classification texts)
- Vision: ViT
- Audio: WavLM-base+ (improved stability over Wav2Vec)

## Results

| Method | Acc (mean) | Acc (max) | vs Baseline |
|--------|-----------|-----------|-------------|
| HVGuard (baseline, uses their CoT) | 0.8456 | 0.8512 | — |
| **AppraiseHate (ours, no baseline CoT)** | **0.8677** | **0.8930** | **+2.2%** |

## Paper Framing Justification

Supported by:
- Chen & Wang (EMNLP 2025): theory guides construct selection, implementation is model-adapted
- Zhao & Daumé (EMNLP 2025): explanation/prediction inconsistency → derive scores from text
- NAACL 2024: use-mention distinction reduces false positives
- ARGUS (CVPR 2025): grounded reasoning
- TCMax (ICLR 2026): modality competition awareness
