# Standard Data Preprocessing & Training Pipeline

## Step 1: Frame Extraction
```bash
python datasets/video_slicer.py -i <video_folder> -o <dataset>/frames --num_frames 32
```
- Script: `datasets/video_slicer.py`
- 32 uniformly sampled frames per video (OpenCV + PyAV fallback)
- Output: `<dataset>/frames/<video_id>/frame_001.jpg ... frame_032.jpg`

## Step 2: Build Quad Images
```bash
python build_quad.py
```
- Script: `build_quad.py`
- Combines every 4 consecutive frames into 2×2 quad images
- Reads from `<dataset>/frames/`, outputs to `<dataset>/quad/`

## Step 3: Audio Extraction
```bash
python datasets/video_to_audio.py -i <video_folder> -o <dataset>/audios
```
- Script: `datasets/video_to_audio.py`
- ffmpeg, 16kHz mono, PCM WAV
- Output: `<dataset>/audios/<video_id>.wav`

## Step 4: Text Embedding
```bash
python embeddings/text_embedding.py --json <dataset>/annotation(new).json --out <emb_dir>/text_features.pth --field text
```
- Script: `embeddings/text_embedding.py`
- BERT-base-uncased [CLS], max_length=128 → 768d

## Step 5: Frame Embedding
```bash
python embeddings/frames_embedding.py --json <dataset>/annotation(new).json --out <emb_dir>/frame_features.pth
```
- Script: `embeddings/frames_embedding.py`
- ViT-base-patch16-224 pooler_output, frame_interval=2, mean pool → 768d

## Step 6: Audio Embedding
```bash
python gen_wavlm_features.py --dataset_name <name>
```
- Script: `gen_wavlm_features.py`
- WavLM-base+ (`microsoft/wavlm-base-plus`), mean pool over time → 768d
- Truncate to 30s max

## Step 7: LLM Prompting
```bash
python AppraiseHate_v13b.py --dataset_name <name> [--language <lang>] --max_concurrent 10
```
- Script: `AppraiseHate_v13b.py` (Harmful/Normal prompt for MHClip & ImpliHateVid)
- Script: `AppraiseHate_v13.py` (Hateful/Non-hateful prompt for HateMM only)
- Model: gpt-5.4-nano, single call, P2C-CoT
- Output: `<dataset>/appraise_v13b_data.json`

## Step 8: LLM Answer Field Embeddings
```bash
python gen_v13b_embeddings.py --dataset_name <name> [--language <lang>]
```
- Script: `gen_v13b_embeddings.py`
- BERT-base-uncased [CLS] encode each answer field (what/target/where/why/how) → 5 × 768d
- Output: `<emb_dir>/v13b_ans_{what,target,where,why,how}_features.pth`

## Step 9: Training & Seed Search
```bash
python run_v13_seed_search.py --dataset_name <name> --num_runs 200 --seed_offset <offset>
```
- Script: `run_v13_seed_search.py`
- Config: C_perfield (text + audio + frame + ans_what + ans_target + ans_where + ans_why + ans_how)
- Fusion: Multi-Head Gated Routing (4 heads, h=192, modality dropout 0.15), no struct
- Training: AdamW lr=2e-4, wCE 1:1.5, label smoothing 0.03, EMA 0.999, 45 epochs
- Retrieval sweep: whitening (none/zca/spca_r32/48/64) × kNN (cosine/csls) × k × temp × alpha + threshold tuning
- 3 offsets (0, 100000, 500000) × 200 seeds = 600 seeds per dataset

## Dataset-Specific Settings

| Dataset | LLM Prompt | Label Mapping | ver |
|---------|-----------|---------------|-----|
| HateMM | v13 (Hateful/Non-hateful) | Non Hate=0, Hate=1 | v13 |
| MHClip-Y | v13b (Harmful/Normal) | Normal=0, Offensive=1, Hateful=1 | v13b |
| MHClip-B | v13b (Harmful/Normal) | Normal=0, Offensive=1, Hateful=1 | v13b |
| ImpliHateVid | v13b (Harmful/Normal) | Normal=0, Hateful=1 | v13b |

## Baseline Scripts
- HVGuard: `baselines/run_hvguard.py --dataset_name <name>`
- ImpliHateVid: `baselines/run_implihatevid.py --dataset_name <name>`
- MoRE: `baselines/MoRE/src/main.py --config-name <config>`
