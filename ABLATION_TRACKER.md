# Ablation Experiment Tracker

## Running (2026-03-19)
- Screen: abl_hmm → logs/ablation_HateMM.log
- Screen: abl_en → logs/ablation_EN.log
- Screen: abl_zh → logs/ablation_ZH.log

## Experiments per dataset (5 seeds each: 42, 1042, 2042, 3042, 4042)

### 1. Full Model
- AppraiseHate (full) — 6mod + wCE1.5 + whiten + kNN

### 2. Staged Ablation
- Raw Fusion (text+audio+frame) — no LLM at all
- + Evidence ledger (text+audio+frame+ev)
- + Relational meaning (text+audio+frame+ev+t1)
- + Alternative appraisals (text+audio+frame+ev+t1+t2) = full 6mod
- + kNN (no whitening)
- + Whiten + kNN = full model

### 3. Field Ablation (remove one at a time, no retrieval)
- Full 6mod (no retrieval) — baseline for field ablation
- - Evidence (text+audio+frame+t1+t2)
- - Relational meaning (text+audio+frame+t2+ev)
- - Alternative appraisals (text+audio+frame+t1+ev)
- - All LLM fields (text+audio+frame)

### 4. 1-Call vs 2-Call
- 1-Call (v9 prompt: v9t1+v9t2, no evidence)
- 2-Call (v12 prompt: t1+t2, no evidence)
- 2-Call (v12 prompt: t1+t2+ev, full)

### 5. Calibration/Retrieval Comparison
- Head only (no kNN)
- + Raw kNN (no whiten)
- + Whiten + kNN

## Experiments NOT included (need new LLM generations)
- Generic CoT baseline
- Generic 2-Call baseline
- Moderation Schema baseline
- Contrastive Schema baseline
→ These require running LLM with different prompts. Track separately.

## Status
- [RUNNING] 3 datasets × 15 configs × 5 seeds = 225 training runs
- Estimated completion: ~40 min per dataset
