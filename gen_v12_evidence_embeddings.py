"""Generate enriched T1 embedding: implicit_meaning + Call 1 evidence combined.

T1E = BERT encode (implicit_meaning + visual_content + spoken_content + tone_and_framing + key_cues)
This gives a richer text embedding than implicit_meaning alone.
"""
import json, sys, os, torch, argparse
sys.path.insert(0, "./embeddings")
from text_embedding import Text_Model

device = "cuda"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", default="HateMM", choices=["HateMM", "Multihateclip"])
    parser.add_argument("--language", default="English")
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

    t1e_features = {}  # enriched T1: implicit + evidence
    evidence_features = {}  # evidence only (separate branch)

    for i, item in enumerate(data):
        vid = item["Video_ID"]
        samples = item.get("v12_samples", [])
        s0 = samples[0] if samples else {}
        ev = s0.get("evidence", {})

        # Enriched T1: implicit_meaning + all evidence fields
        im = s0.get("implicit_meaning", "") or ""
        visual = ev.get("visual_content", "") or ""
        spoken = ev.get("spoken_content", "") or ""
        tone = ev.get("tone_and_framing", "") or ""
        cues = ev.get("key_cues", [])
        if isinstance(cues, list):
            cues = ", ".join(str(c) for c in cues)

        t1e_text = f"{im} Visual: {visual} Spoken: {spoken} Tone: {tone} Cues: {cues}"
        t1e_features[vid] = model(t1e_text or " ").to(device)

        # Evidence only
        ev_text = f"Visual: {visual} Spoken: {spoken} Tone: {tone} Cues: {cues}"
        evidence_features[vid] = model(ev_text or " ").to(device)

        if (i + 1) % 200 == 0:
            print(f"Processed {i+1}/{len(data)}")

    torch.save(t1e_features, f"{emb_dir}/v12_t1e_features.pth")
    torch.save(evidence_features, f"{emb_dir}/v12_evidence_features.pth")
    print(f"Saved: T1E={len(t1e_features)}, Evidence={len(evidence_features)}")


if __name__ == "__main__":
    main()
