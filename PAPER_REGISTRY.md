# Paper Registry

## Prompt Design (Step 1)

| Paper | Venue | Year | Key Finding | Used in our work? | Worked? |
|-------|-------|------|------------|-------------------|---------|
| Beyond Context to Cognitive Appraisal | ACL Findings | 2025 | CAT for emotion reasoning via appraisal dimensions | Theory foundation | N/A |
| Third-Person Appraisal Agent | EMNLP Findings | 2025 | 3-phase appraisal: Primary→Secondary→Reappraisal | Inspired v6/v7 3-phase chain | v6/v7 worse than v5 |
| NLP Systems Can't Tell Use from Mention | NAACL | 2024 | Use-mention distinction reduces FP by 82.61% | **Inspired v9 stance field** | **Yes (+1.5%)** |
| Structured templates > free-form CoT (Dang et al.) | EMNLP | 2025 | Structured output beats verbose CoT | Informed v5 design | Yes |
| ARGUS: Grounded CoT | CVPR | 2025 | Ground reasoning in visual evidence | Informed grounding approach | Partially |
| VQAGuider: Decompose video QA | ACL | 2025 | Atomic sub-tasks improve video QA | Informed decomposition | Partially |
| Explanation/prediction inconsistency (Zhao & Daumé) | EMNLP | 2025 | LLM explanations can be unfaithful to predictions | Identified our score-text bug | Yes (v6 fix attempt) |
| Theory-grounded measurement (Faulborn et al.) | ACL | 2025 | Prompt analysis needs theory-grounded measurement | Supports CAT framing | N/A |
| Chen & Wang | EMNLP | 2025 | Theory guides construct, model adapts implementation | **Justifies v9 approach** | **Yes** |
| Liu et al. | ACL | 2025 | Simpler prompting can outperform complex | Justifies v5/v9 simplicity | Yes |
| Wang et al. | ICML | 2025 | Unintuitive prompt forms can work | Supports empirical tuning | N/A |

## Fusion Design (Step 2)

| Paper | Venue | Year | Method | Used? | Worked? |
|-------|-------|------|--------|-------|---------|
| TCMax | ICLR | 2026 | Total Correlation Maximization for modality competition | **Tried** | Max improved, mean unchanged |
| MIAM | ICLR | 2026 | Adaptive modality masking | Inspired asymm dropout | Marginal |
| CREMA | ICLR | 2025 | Progressive modular fusion + LoRA | Tried simplified | No (worse than AC-MHGF) |
| AUG (Rethinking Multimodal Learning) | NeurIPS | 2025 | Mitigate classification ability disproportion | Not tried | — |
| Representation Collapse | ICML | 2025 | Models rely on subset of modalities | Design guidance | N/A |
| CyIN | NeurIPS | 2025 | Cyclic informative latent space | Not tried (missing modality focus) | — |
| AVQACL (QCIF) | CVPR | 2025 | Question-guided cross-modal fusion | Not tried | — |

## Encoder (Step 3)

| Encoder | Type | Year | Tried? | Result |
|---------|------|------|--------|--------|
| BERT-base | Text | 2018 | Yes | **Best text encoder** |
| DeBERTa-v3-base | Text | 2021 | Yes | Slightly worse |
| DeBERTa-v3-large | Text | 2021 | Yes | Worse |
| ModernBERT-base | Text | 2024 | Yes | Much worse |
| GTE-ModernBERT | Text | 2025 | Yes | Worse |
| ViT | Vision | 2021 | Yes | **Best vision encoder** |
| SigLIP2-base | Vision | 2025 | Yes | Worse |
| Wav2Vec2-base | Audio | 2020 | Yes | Baseline |
| **WavLM-base+** | Audio | 2022 | **Yes** | **Better stability** |
| Qwen3-Embedding-8B | Text | 2025 | Not tried (8B too large for frozen) | — |
| EmbeddingGemma | Text | 2025 | Not tried | — |

## Hate Detection Domain

| Paper | Venue | Year | Key Finding |
|-------|-------|------|------------|
| HVGuard | EMNLP | 2025 | Our baseline, CoT + MoE, reports ~0.86 acc |
| MM-HSD | ACM MM | 2025 | OCR as query modality |
| MultiHateLoc | arXiv | 2025 | Temporal localization for hate videos |
| Text Takes Over | EMNLP | 2025 | Text dominates in multimodal hate detection |
| HATEDAY | ACL | 2025 | Global hate speech dataset |
| BOLT | CVPR | 2025 | Query-aware frame selection > uniform |
