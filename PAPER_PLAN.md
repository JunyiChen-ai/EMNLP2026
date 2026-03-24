# AppraiseHate Paper Plan

## Method Structure (3 modules)

### Module 1: CAT-Guided Dual-Stage LLM Analysis
- Call 1 (Evidence Extraction): LLM sees quad images → outputs evidence ledger (visual_content, spoken_content, tone_and_framing, key_cues)
- Call 2 (Appraisal Judgment): LLM sees only Call 1 text → outputs relational_meaning, alternative_appraisals, speaker_stance, target_group
- Theoretical basis: CAT's stimulus encoding → cognitive appraisal process
- Key field naming:
  - implicit_meaning → **relational_meaning** (Lazarus's core relational meaning)
  - contrastive_readings → **alternative_appraisals** (same stimulus, different appraisals)

### Module 2: Appraisal-Conditioned Multi-Modal Fusion
- 6 modalities (768d each): text, audio, frame, relational_meaning (T1), alternative_appraisals (T2), evidence
- Multi-Head Gated Routing Fusion (4 heads, h=192, modality dropout 0.15)
- struct (9d) removed — not useful enough
- Trained with CE + label smoothing 0.03

### Module 3: Case-Based Retrieval-Augmented Inference
- After training, extract 64d penultimate features from fusion
- Feature whitening (ZCA/shrinkage-PCA) — geometric conditioning for meaningful similarity
- kNN retrieval: find k nearest training neighbors, similarity-weighted vote
- Blend: final_logits = (1-α)·head_logits + α·kNN_logits
- Justification: mirrors content moderation practice (learn rules during training, consult precedents during practice)
- Theoretical basis: schema learning (training) + episodic retrieval under ambiguity (inference)

## Naming Conventions
- implicit_meaning → relational_meaning (Lazarus)
- contrastive_readings → alternative_appraisals
- evidence ledger → evidence ledger (no change)
- kNN retrieval → case-based retrieval
- whitening → geometric conditioning / similarity-space conditioning

## What NOT to write
- Weighted CE 1:1.5 (engineering trick)
- Threshold tuning (standard calibration)
- Seed search (internal optimization)
- Whitening as cognitive process (it's a technical prerequisite)

## Key Citations for Story
- Lazarus 1991: CAT, relational meaning
- Scherer 2009: Component Process Model, appraisal checks
- Nosofsky 1986: GCM, exemplar-based categorization
- Clarke et al. ACL 2023: Rule By Example (rules + exemplars for moderation)
- Klonick 2018: moderator training = rules, practice = precedents
- Tullis et al. 2014: episodic recall aids ambiguous judgment
- Brod et al. 2017: schemas as generalized abstractions

## Literature Landscape

### Non-LLM Methods (traditional multimodal fusion)
- HateMM baseline (Das et al., ICWSM 2023): BERT+ViT+MFCC → LSTM → late fusion, acc~0.80
- HateCLIPper: fine-tune CLIP projections for text-image alignment
- Mei et al. 2023: CLIP + contrastive alignment + dynamic retrieval
- Zhang et al. 2024: complex cross-modal attention fusion
- MultiHateGNN (Yue et al., BMVC 2025): GNN on multimodal features
- CMFusion (2025): channel-wise + modality-wise fusion + temporal cross-attention
- MM-HSD (Céspedes-Sarrias et al., 2025): concat transcript+audio+video+OCR + cross-modal attention, SOTA F1=0.874
- TANDEM (2026): temporal-aware neural detection

### LLM-based Methods
- HVGuard (EMNLP 2025): 3-step CoT (describe frames→analyze text→combine) + MoE fusion
- RAMF (arXiv 2025): adversarial reasoning (objective + hate-assumed + non-hate-assumed)
- Training-free detection (arXiv 2026): multi-stage adversarial reasoning, no training
- Hateful meme LMM series (CVPR/WWW/EMNLP 2024-2025): GPT-4V/mPLUG-Owl explainable detection

### Identified Limitations (for intro §3)
1. Surface-level fusion: features are concatenated/attended, not interpreted for meaning
2. Atheoretical LLM reasoning: HVGuard/RAMF CoT is ad-hoc engineering, no principled design
3. Parametric fragility on ambiguous samples: small data + inherent ambiguity → overconfident FN

## Intro Structure

### §1: Research background
- Online hate spreading via video, multimodal challenge

### §2: Existing methods
- Non-LLM: feature extraction + fusion (HateMM, HateCLIPper, MM-HSD, CMFusion, MultiHateGNN)
- LLM-based: CoT reasoning (HVGuard, RAMF, training-free)

### §3: Limitations/Challenges (most important)
1. Surface fusion misses interpretive meaning
2. LLM reasoning lacks theoretical grounding → shortcut-prone
3. Parametric classifier fragile on ambiguous/implicit hate in small-data regime

### §4: Our method addresses each challenge
1. → CAT-guided dual-stage LLM (evidence→appraisal, structured fields with theoretical basis)
2. → Each output field grounded in appraisal theory (relational_meaning, alternative_appraisals)
3. → Case-based retrieval augmentation at inference (mirrors moderation practice)

### §5: Contributions
1. CAT-grounded appraisal framework for hateful video detection
2. Case-based retrieval-augmented inference for ambiguous samples
3. Comprehensive evaluation on 3 datasets (HateMM, EN/ZH MultiHateClip)

## Controlled Internal Baselines (need new LLM generations)

### 1. Raw Fusion
- text+audio+frame only, no LLM
- Status: [RUNNING in ablation experiments, no LLM needed]

### 2. Generic CoT
- Single-call LLM, free-form chain-of-thought ("describe this video and judge if hateful")
- Same LLM (gpt-4.1-nano), same fusion architecture
- Need: new LLM generation with generic CoT prompt → BERT encode → train fusion
- Status: [TODO — need LLM API call]

### 3. Generic 2-Call
- Two-call LLM, free-form (Call 1: describe content, Call 2: judge)
- No structured fields, no CAT
- Same LLM, same fusion
- Need: new LLM generation → BERT encode → train fusion
- Status: [TODO — need LLM API call]

### 4. Moderation Schema
- Two-call LLM with moderation-aligned structured outputs
- Fields: target_group, speaker_stance, intent/severity, content_summary
- NOT CAT-grounded, but task-relevant structured prompting
- Purpose: prove that structured prompting alone doesn't match CAT-specific fields
- Need: new LLM generation → BERT encode → train fusion
- Status: [TODO — need LLM API call]

### 5. Contrastive Schema (strongest non-CAT control)
- Two-call LLM with fields designed to MATCH our structure without CAT motivation
- Fields: primary_interpretation, plausible_alternative_interpretation, speaker_stance, target_group, severity
- Same 2-call budget, same fusion, same field count
- Purpose: isolate whether CAT-specific constructs (relational meaning, alternative appraisals) provide value beyond equally-structured atheoretical prompting
- GPT says: "If AppraiseHate still wins, the CAT claim becomes much harder to dismiss"
- Need: new LLM generation → BERT encode → train fusion
- Status: [TODO — need LLM API call]

### Why these baselines matter (GPT warning)
- Without controlled baselines, reviewers will say "gain comes from more LLM text, not CAT"
- Contrastive Schema is the KEY baseline — it's the strongest possible non-CAT control
- If we beat it by 3%+, the CAT story is validated

## GPT Feedback on Intro (Round 1)

### Core identity
> "Hateful video detection is an appraisal problem before it is a fusion problem."

### Compress §3 from 3 challenges to 2:
1. **Interpretive challenge**: hateful meaning is not directly observed from surface features; it emerges from appraised meaning, stance, and target-sensitive interpretation. Neither fusion methods nor LLM CoT explicitly model the intermediate judgments that determine hate.
2. **Decision challenge**: even with structured interpretation, hate judgments are ambiguous and context-dependent. Same content can be endorsement/satire/counterspeech/reporting. Global parametric classifiers are brittle on these edge cases.

### §3→§4 mapping:
- Interpretive challenge → CAT-guided evidence→appraisal pipeline
- Decision challenge → Case-based retrieval refinement

### Recommended 5-paragraph structure:
1. Hateful videos are multimodal, implicit, socially harmful
2. Prior work (fusion + LLM reasoning) stops at "what is in the video" not "how content is appraised"
3. Hate judgments are ambiguous, context-sensitive; labels shift with context; low-resource → brittle boundaries
4. Our method: CAT-guided appraisal + case-based refinement
5. Contributions

### Key sentence to use:
> "Existing multimodal detectors are good at representing what a video contains, but hateful video classification often depends on how that content is appraised: whether it attributes blame, implies threat, dehumanizes a target group, or merely reports or contests such views."

### What to AVOID:
- "Existing LLM methods lack theoretical grounding" — too accusatory, sounds like justifying CAT
- Better: "current methods do not explicitly model the intermediate appraisals that determine hate"

### Missing related work to cite:
- Policy-grounded moderation: Mullick+ ACL 2023, HateModerate Findings NAACL 2024, Policy-as-Prompt FAccT 2025
- Rules+examples in moderation: Clarke+ ACL 2023 (Rule By Example)
- Context/subjectivity in hate: Yu+ NAACL 2022, Ljubešić+ 2022, HateWiC EMNLP 2024
- CAT in NLP: AppraisePLM CoNLL 2025, Third-Person Appraisal Agent EMNLP Findings 2025

## Experiments Plan

### Metrics
- Primary: Macro-F1
- Secondary: Accuracy
- Tertiary: Hate-class Recall / F1
- All results: mean ± std over 3-5 seeds

### Main Text Experiments (by priority)

#### 1. Experimental Setup
- Datasets, splits, metrics, seed protocol
- Same accessible-video subset for all baselines

#### 2. Main Results Table (two baseline blocks)
**Published baselines:**
- HateMM paper baselines (Das et al. 2023)
- MultiHateClip paper baselines (Wang et al. 2024)
- HVGuard (EMNLP 2025)
- ImpliHateVid (ACL 2025) on HateMM
- MM-HSD / CMFusion / MultiHateGNN (if reproducible)

**Controlled internal baselines (CRITICAL — GPT says reviewers will require):**
- Raw-media fusion only (text+audio+frame, no LLM)
- Direct MLLM judge (LLM直接判hateful/not)
- Generic 1-call CoT + fusion (same LLM, same fusion, but no CAT structure)
- Generic 2-call reasoning + fusion (same architecture, but free-form instead of CAT-guided)

#### 3. Claim-Oriented Core Ablation (one compact table)
Organized as staged additions:
- Raw fusion (text+audio+frame only)
- + generic LLM summary text
- + evidence ledger
- + relational_meaning
- + alternative_appraisals
- 1-call CAT vs 2-call evidence→appraisal
- + kNN only (no whitening)
- + whitening + kNN

#### 4. CAT Analysis (strengthens theoretical claim)
Must-have:
- 2-call vs 1-call comparison
- CAT-structured outputs vs free-form CoT with similar token budget
- Field ablation: relational_meaning, alternative_appraisals, evidence individual contribution
Strong extra:
- Evaluate LLM target-group accuracy against MultiHateClip annotations
- Evaluate evidence overlap with HateMM hate-span annotations

#### 5. Retrieval Analysis (strengthens decision-challenge claim)
Must-have:
- head-only vs kNN-only vs interpolation
- Gains by confidence/entropy bin (show kNN helps uncertain samples most)
- ECE or Brier score before vs after retrieval
- False-negative rescue rate on hate class
Strong extra:
- Compare against threshold tuning and temperature scaling alone (prove it's not just calibration)
- Training-time retrieval baseline (prove inference-only is better)

#### 6. Qualitative Case Studies (2-3 cases)
Show: evidence ledger → relational_meaning → alternative_appraisals → top kNN neighbors → classifier vs refined prediction

### Appendix Experiments
- Full unimodal baselines
- Detailed field/modality ablations
- Hyperparameter sensitivity (k, alpha, whitening rank)
- Retrieval space variants (whitened vs raw, cosine vs CSLS)
- Training-time retrieval baseline (if implemented)
- LLM cost/runtime/token budget
- Per-class P/R/F1 and confusion matrices
- More qualitative examples and failure cases
- Seed-by-seed results

### Organization Principle
Every experiment must support one of three claims:
1. Hate detection requires appraised meaning, not just surface fusion
2. CAT-guided outputs are better than generic LLM reasoning
3. Case-based refinement helps ambiguous decisions
If it doesn't → appendix.

### GPT Warnings
- MUST have controlled internal baselines (same LLM, same fusion, no CAT) — otherwise reviewers say gain is from "more LLM text"
- MUST compare kNN against simple calibration (temperature scaling, threshold tuning) — otherwise kNN looks like calibration trick
- Organize ablations by CLAIM not by module — avoid looking like hyperparameter tuning
