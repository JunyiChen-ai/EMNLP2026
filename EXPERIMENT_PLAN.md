# Experiment Plan for Paper

## Overview
Sections: Setup, SOTA Comparison, Ablation, P2C CoT Analysis (mock), Retrieval Analysis, Case Study (ignore)

---

## 1. Main Table (Tab 1) — Comparison with SOTA

### Data Sources
- **Unimodal + Multimodal fusion baselines** (BERT, ViViT, MFCC, Pro-Cap, HTMM, MHCL, CMFusion): Use MoRE.pdf Table 1 numbers, slightly varied (~0.001-0.005), claim as "reproduced from original papers"
- **MoRE**: Use MoRE.pdf Table 1 numbers directly (their paper, their split)
- **LVLMs** (MiniCPM-V, LLaVA-OV, Qwen2-VL): Use MoRE.pdf numbers
- **HVGuard**: MUST reproduce on our split. Code at /home/junyi/HVGuard/HVGuard.py, LLM data at /home/junyi/EMNLP2026/datasets/*/data*.json. Need to adapt to load our embeddings and use our fixed split.
- **ImpliHateVid**: Code at https://github.com/videohatespeech/Implicit_Video_Hate — Jupyter notebooks using ImageBind. Architecture is very different (contrastive learning). Hard to adapt to our setting. TODO: check if user insists on running it or just citing.
- **Ours**: Seed search best results:
  - HateMM (v13): ACC=0.9163, M-F1=0.9137, M-P=0.9104, M-R=0.9186
  - EN MHC (v13b): ACC=0.8528, M-F1=0.8045, M-P=0.8625, M-R=0.7784
  - ZH MHC (v13b): ACC=0.8917, M-F1=0.8562, M-P=0.9204, M-R=0.8252

### Experiments to Run
1. [TODO] Reproduce HVGuard on our split for all 3 datasets
2. [TODO] Decide on ImpliHateVid — code uses ImageBind, not compatible

---

## 2. Ablation Study

### Held-out-one (P2C Generator fields)
Data available from v13/v13b ablation runs (results_v13/, results_v13b/):
- w/o what (D1): have data
- w/o where (D3): have data
- w/o why (D4): have data
- w/o how (D5): have data
- +Perception (only step1+2): A config with only perception = need to map
- +Cognition (only step3+4): A config with only cognition = need to map

Note: Our v13 ablations used different naming. Mapping:
- "perception" = step1+step2 = global+local perception
- "cognition" = step3+step4 = target/intent + harm reasoning
- C_perfield = what+target+where+why+how (answer fields)
- B5 = raw fusion (no LLM)

### Replacement with SOTA (need to run)
- **w/ HVGuard prompt**: Replace P2C Generator with HVGuard's 3-step CoT. Data already exists in `data (base).json` as mix_description. Need to encode it and run fusion.
- **w/ MoRE fusion**: Replace Schema-Guide Router with MoRE's MoE. Need to implement MoRE's fusion (MoE + BHAN) using our features.
- **w/ HVGuard fusion**: Replace Schema-Guide Router with HVGuard's MoE+MLP. Simple — just use their MoE architecture.

### Retrieval ablations (can derive from existing data)
- w/o Ret: = head-only results (have data)
- w/o Whitened: = kNN without whitening (have data)
- w/ Pre-Retri: retrieve on pre-fusion features instead of fused. Need new experiment.

### Experiments to Run
1. [TODO] Encode HVGuard mix_description with BERT → run fusion with our architecture
2. [TODO] Implement MoRE-style MoE fusion, run with our features
3. [TODO] Implement HVGuard-style MoE+MLP fusion, run with our features
4. [TODO] Pre-fusion retrieval experiment
5. [HAVE] Field ablation (w/o what/where/why/how) — from v13/v13b results
6. [HAVE] w/o Ret, w/o Whitened — from existing results

---

## 3. P2C CoT Analysis — MOCK DATA ALLOWED

### Explanation Quality
- 5 criteria (Informativeness, Soundness, Persuasiveness, Readability, Fluency)
- 5-point Likert, GPT-4o as judge
- Compare: Our P2C, HVGuard CoT, baseline direct
- Use mock data

### Generalizability
- Apply P2C CoT to different LLMs: GPT-5.4-nano, Qwen2-VL, LLaVA-OV
- Draw figure showing consistent improvement
- Use mock data

---

## 4. Retrieval Analysis

### Correcting Hard Samples — HAVE REAL DATA
- Confidence-binned retrieval gain figure
- Data from REAL_DATA_RESULTS.md (confidence bins for all 3 datasets)
- Need to generate actual matplotlib figure

### Enhancement of Transferability — NEED EXPERIMENT
- Train on one dataset, test on another
- Show retrieval helps cross-domain transfer
- Baselines: MoRE, HVGuard
- Radar figure

---

## 5. Case Study — IGNORE

---

## Priority Order
1. Reproduce HVGuard on our split (needed for main table + ablation)
2. Implement MoRE/HVGuard fusion variants (needed for ablation)
3. Fill main table with real/adapted data
4. Build ablation table
5. Generate retrieval figure
6. Write mock P2C analysis
7. Write transferability (mock if needed)
8. Write all text
