"""Ablation: Replace our T1 (implicit_meaning) embedding with HVGuard's Mix_description embedding.

This tests whether the performance gap on MultiHateClip is due to our LLM's implicit meaning quality.

Steps:
1. Encode HVGuard's Mix_description with BERT-base → hvguard_t1_features.pth
2. Run main.py using hvguard_t1 instead of our v9_t1
"""
import json, sys, os, torch, argparse
sys.path.insert(0, "./embeddings")
from text_embedding import Text_Model

device = "cuda"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", required=True, choices=["English", "Chinese"])
    args = parser.parse_args()

    lang = args.language
    emb_dir = f"./embeddings/Multihateclip/{lang}"
    os.makedirs(emb_dir, exist_ok=True)

    # Load HVGuard's data.json
    with open(f"./datasets/Multihateclip/{lang}/hvguard_data.json") as f:
        hv_data = json.load(f)

    model = Text_Model()  # BERT-base [CLS] max_len=128
    hvguard_t1 = {}

    for i, item in enumerate(hv_data):
        vid = item["Video_ID"]
        mix = item.get("Mix_description", "") or " "
        hvguard_t1[vid] = model(mix).to(device)
        if (i + 1) % 200 == 0:
            print(f"Processed {i+1}/{len(hv_data)}")

    torch.save(hvguard_t1, f"{emb_dir}/hvguard_t1_features.pth")
    print(f"Saved: {len(hvguard_t1)} items to {emb_dir}/hvguard_t1_features.pth")


if __name__ == "__main__":
    main()
