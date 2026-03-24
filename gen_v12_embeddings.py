"""Generate v12 embeddings from 2-call pipeline output.

T1: implicit_meaning → BERT 768d
T2: contrastive_readings → BERT 768d
Struct: speaker_stance one-hot + target_present + multi-sample stats
No appraisal_vector — CAT is in the reasoning, not the output.
"""
import json, sys, os, torch, numpy as np, argparse
from collections import Counter
sys.path.insert(0, "./embeddings")
from text_embedding import Text_Model

device = "cuda"
STANCES = ["endorse", "report", "condemn", "mock", "unclear"]


def extract_struct(samples):
    if not samples:
        return torch.zeros(len(STANCES) + 4)
    all_stances = []
    has_target = []
    for s in samples:
        stance = s.get("speaker_stance", "unclear").strip().lower()
        all_stances.append(STANCES.index(stance) if stance in STANCES else 4)
        t = s.get("target_group", "")
        has_target.append(1.0 if t and t.lower() not in ("none", "n/a", "", "no specific group") else 0.0)

    # Stance one-hot (majority)
    stance_oh = [0.0] * len(STANCES)
    mc = Counter(all_stances).most_common(1)[0][0]
    stance_oh[mc] = 1.0

    features = list(stance_oh)
    features.append(np.mean(has_target))
    # Stance agreement
    features.append(1.0 if len(set(all_stances)) == 1 else 0.0)
    # Endorse ratio
    features.append(sum(1 for s in all_stances if s == 0) / len(all_stances))
    features.append(float(len(samples)))
    return torch.tensor(features, dtype=torch.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", default="HateMM", choices=["HateMM", "Multihateclip"])
    parser.add_argument("--language", default="English", choices=["English", "Chinese"])
    args = parser.parse_args()

    if args.dataset_name == "HateMM":
        data_path = "./datasets/HateMM/appraise_v12_data.json"
        emb_dir = "./embeddings/HateMM"
    else:
        data_path = f"./datasets/Multihateclip/{args.language}/appraise_v12_data.json"
        emb_dir = f"./embeddings/Multihateclip/{args.language}"
    os.makedirs(emb_dir, exist_ok=True)

    model = Text_Model()
    with open(data_path) as f:
        data = json.load(f)

    t1_features, t2_features, struct_features = {}, {}, {}

    for i, item in enumerate(data):
        vid = item["Video_ID"]
        samples = item.get("v12_samples", [])
        s0 = samples[0] if samples else {}

        # T1: implicit_meaning
        t1_text = s0.get("implicit_meaning", "") or " "
        t1_features[vid] = model(t1_text).to(device)

        # T2: contrastive_readings
        cr = s0.get("contrastive_readings", {})
        t2_text = f"Hateful: {cr.get('hateful', '')} Non-hateful: {cr.get('non_hateful', '')}"
        t2_features[vid] = model(t2_text or " ").to(device)

        # Struct
        struct_features[vid] = extract_struct(samples).to(device)

        if (i + 1) % 200 == 0:
            print(f"Processed {i+1}/{len(data)}")

    torch.save(t1_features, f"{emb_dir}/v12_t1_features.pth")
    torch.save(t2_features, f"{emb_dir}/v12_t2_features.pth")
    torch.save(struct_features, f"{emb_dir}/v12_struct_features.pth")
    sd = struct_features[list(struct_features.keys())[0]].shape[0]
    print(f"Saved: T1={len(t1_features)}, T2={len(t2_features)}, struct(dim={sd})={len(struct_features)}")


if __name__ == "__main__":
    main()
