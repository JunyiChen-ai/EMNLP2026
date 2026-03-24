"""Generate v10 embeddings from AppraiseHate v10 outputs."""

import json
import os
import sys
from collections import Counter

import numpy as np
import torch

sys.path.insert(0, "./embeddings")
from text_embedding import Text_Model

device = "cuda"
APPRAISAL_KEYS = [
    "blame",
    "threat",
    "disgust",
    "dominance",
    "exclusion",
    "dehumanization",
    "harm_legitimization",
]
STANCES = ["endorse", "report", "condemn", "mock", "unclear"]
NULL_TARGETS = {"", "none", "no group", "n/a", "na", "unclear", "unknown"}


def _target_present(value):
    target = (value or "").strip().lower()
    return 0.0 if target in NULL_TARGETS else 1.0


def extract_struct(samples):
    if not samples:
        return torch.zeros(7 * 4 + len(STANCES) + 2 + 3)

    per_key = {k: [] for k in APPRAISAL_KEYS}
    stance_ids = []
    quoted_vals = []
    target_vals = []

    for sample in samples:
        av = sample.get("appraisal_vector", {})
        for key in APPRAISAL_KEYS:
            value = av.get(key, 0)
            per_key[key].append(float(value) if isinstance(value, (int, float)) else 0.0)
        stance = (sample.get("stance", "unclear") or "unclear").strip().lower()
        stance_ids.append(STANCES.index(stance) if stance in STANCES else 4)
        quoted_vals.append(1.0 if sample.get("quoted_or_criticized", False) else 0.0)
        target_vals.append(_target_present(sample.get("target_group", "")))

    features = []
    for key in APPRAISAL_KEYS:
        vals = per_key[key]
        features.extend([np.mean(vals), np.std(vals), np.min(vals), np.max(vals)])

    stance_oh = [0.0] * len(STANCES)
    stance_oh[Counter(stance_ids).most_common(1)[0][0]] = 1.0
    features.extend(stance_oh)

    features.append(float(np.mean(quoted_vals)))
    features.append(float(np.mean(target_vals)))

    totals = [sum(per_key[key][i] for key in APPRAISAL_KEYS) for i in range(len(samples))]
    features.append(np.mean(totals) / 14.0)
    features.append(np.std(totals) / 14.0)
    features.append(float(len(samples)))
    return torch.tensor(features, dtype=torch.float32)


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="HateMM",
        choices=["HateMM", "Multihateclip"],
    )
    parser.add_argument(
        "--language",
        type=str,
        default="English",
        choices=["English", "Chinese"],
    )
    args = parser.parse_args()

    if args.dataset_name == "HateMM":
        data_path = "./datasets/HateMM/appraise_v10_data.json"
        emb_dir = "./embeddings/HateMM"
    else:
        data_path = f"./datasets/Multihateclip/{args.language}/appraise_v10_data.json"
        emb_dir = f"./embeddings/Multihateclip/{args.language}"
    os.makedirs(emb_dir, exist_ok=True)

    model = Text_Model()
    with open(data_path) as f:
        data = json.load(f)

    t1_features = {}
    t2_features = {}
    score_features = {}
    struct_features = {}

    for i, item in enumerate(data):
        vid = item["Video_ID"]
        samples = item.get("v10_samples", [])
        sample0 = samples[0] if samples else {}

        target_group = sample0.get("target_group", "none") or "none"
        stance = sample0.get("stance", "unclear") or "unclear"
        quoted = bool(sample0.get("quoted_or_criticized", False))

        implicit = sample0.get("implicit_meaning", "") or " "
        t1_text = (
            f"Target: {target_group}. "
            f"Meaning: {implicit} "
            f"Stance: {stance}. "
            f"Quoted_or_criticized: {quoted}."
        )
        t1_features[vid] = model(t1_text).to(device)

        cr = sample0.get("contrastive_readings", {})
        t2_text = (
            f"Hateful: {cr.get('hateful', '')} "
            f"Non-hateful: {cr.get('non_hateful', '')} "
            f"Stance: {stance}. "
            f"Quoted_or_criticized: {quoted}."
        )
        t2_features[vid] = model(t2_text or " ").to(device)

        av = sample0.get("appraisal_vector", {})
        scores = [
            min(max(av.get(key, 0), 0), 2) / 2.0
            if isinstance(av.get(key, 0), (int, float))
            else 0.0
            for key in APPRAISAL_KEYS
        ]
        score_features[vid] = torch.tensor(scores, dtype=torch.float32, device=device)
        struct_features[vid] = extract_struct(samples).to(device)

        if (i + 1) % 200 == 0:
            print(f"Processed {i + 1}/{len(data)}")

    torch.save(t1_features, f"{emb_dir}/v10_t1_features.pth")
    torch.save(t2_features, f"{emb_dir}/v10_t2_features.pth")
    torch.save(score_features, f"{emb_dir}/v10_scores.pth")
    torch.save(struct_features, f"{emb_dir}/v10_struct_features.pth")
    struct_dim = struct_features[list(struct_features.keys())[0]].shape[0]
    print(
        f"Saved: T1={len(t1_features)}, T2={len(t2_features)}, "
        f"scores={len(score_features)}, struct(dim={struct_dim})={len(struct_features)}"
    )


if __name__ == "__main__":
    main()
