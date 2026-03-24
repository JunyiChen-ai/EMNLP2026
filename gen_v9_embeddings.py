"""Generate v9 embeddings: same as v5 + stance features."""
import json, sys, os, torch, numpy as np
from collections import Counter
sys.path.insert(0, "./embeddings")
from text_embedding import Text_Model

device = "cuda"
APPRAISAL_KEYS = ["blame", "threat", "disgust", "dominance", "exclusion", "dehumanization", "harm_legitimization"]
STANCES = ["endorse", "report", "condemn", "mock", "unclear"]


def extract_struct(samples):
    if not samples:
        return torch.zeros(7 * 4 + len(STANCES) + 3)
    per_key = {k: [] for k in APPRAISAL_KEYS}
    all_stances = []
    for s in samples:
        av = s.get("appraisal_vector", {})
        for k in APPRAISAL_KEYS:
            v = av.get(k, 0)
            per_key[k].append(float(v) if isinstance(v, (int, float)) else 0.0)
        stance = s.get("stance", "unclear").strip().lower()
        all_stances.append(STANCES.index(stance) if stance in STANCES else 4)
    features = []
    for k in APPRAISAL_KEYS:
        vals = per_key[k]
        features.extend([np.mean(vals), np.std(vals), np.min(vals), np.max(vals)])
    # Stance one-hot (majority)
    stance_oh = [0.0] * len(STANCES)
    mc = Counter(all_stances).most_common(1)[0][0]
    stance_oh[mc] = 1.0
    features.extend(stance_oh)
    # Aggregates
    total = [sum(per_key[k][i] for k in APPRAISAL_KEYS) for i in range(len(samples))]
    features.append(np.mean(total) / 14.0)
    features.append(np.std(total) / 14.0)
    features.append(float(len(samples)))
    return torch.tensor(features, dtype=torch.float32)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="HateMM", choices=["HateMM", "Multihateclip"])
    parser.add_argument("--language", type=str, default="English", choices=["English", "Chinese"])
    args = parser.parse_args()

    if args.dataset_name == "HateMM":
        data_path = "./datasets/HateMM/appraise_v9_data.json"
        emb_dir = "./embeddings/HateMM"
    else:
        data_path = f"./datasets/Multihateclip/{args.language}/appraise_v9_data.json"
        emb_dir = f"./embeddings/Multihateclip/{args.language}"
    os.makedirs(emb_dir, exist_ok=True)

    model = Text_Model()
    with open(data_path) as f:
        data = json.load(f)

    t1_features, t2_features, score_features, struct_features = {}, {}, {}, {}

    for i, item in enumerate(data):
        vid = item["Video_ID"]
        samples = item.get("v9_samples", [])
        s0 = samples[0] if samples else {}

        # T1: implicit_meaning (same as v5)
        t1_text = s0.get("implicit_meaning", "") or " "
        t1_features[vid] = model(t1_text).to(device)

        # T2: contrastive_readings (same as v5)
        cr = s0.get("contrastive_readings", {})
        t2_text = f"Hateful: {cr.get('hateful', '')} Non-hateful: {cr.get('non_hateful', '')}"
        t2_features[vid] = model(t2_text or " ").to(device)

        # Scores (same as v5)
        av = s0.get("appraisal_vector", {})
        scores = [min(max(av.get(k, 0), 0), 2) / 2.0 if isinstance(av.get(k, 0), (int, float)) else 0.0 for k in APPRAISAL_KEYS]
        score_features[vid] = torch.tensor(scores, dtype=torch.float32, device=device)

        # Structured (v5 + stance)
        struct_features[vid] = extract_struct(samples).to(device)

        if (i + 1) % 200 == 0:
            print(f"Processed {i+1}/{len(data)}")

    torch.save(t1_features, f"{emb_dir}/v9_t1_features.pth")
    torch.save(t2_features, f"{emb_dir}/v9_t2_features.pth")
    torch.save(score_features, f"{emb_dir}/v9_scores.pth")
    torch.save(struct_features, f"{emb_dir}/v9_struct_features.pth")
    sd = struct_features[list(struct_features.keys())[0]].shape[0]
    print(f"Saved: T1={len(t1_features)}, T2={len(t2_features)}, scores={len(score_features)}, struct(dim={sd})={len(struct_features)}")


if __name__ == "__main__":
    main()
