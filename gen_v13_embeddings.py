"""
Generate v13 embeddings from AppraiseHate v13 LLM outputs.

Group A (combined):
- perception: BERT encode (step1 + step2) -> 768d
- cognition: BERT encode (step3 + step4) -> 768d
- answer: BERT encode (full answer paragraph) -> 768d

Group B (per-field, each 768d):
- ans_which, ans_what, ans_target, ans_where, ans_why, ans_how

struct: 1d (is_hateful flag from LLM)
"""
import argparse, json, os
import torch
from transformers import AutoTokenizer, AutoModel

device = "cuda" if torch.cuda.is_available() else "cpu"

def encode_texts(texts, tokenizer, model, max_length=256):
    """BERT [CLS] encode a list of texts."""
    features = {}
    model.eval()
    with torch.no_grad():
        for vid, text in texts.items():
            if not text or text.strip() == "":
                text = "No content available."
            inputs = tokenizer(text, return_tensors="pt", truncation=True,
                             max_length=max_length, padding=True).to(device)
            outputs = model(**inputs)
            features[vid] = outputs.last_hidden_state[:, 0, :].squeeze(0).cpu()
    return features

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", default="HateMM", choices=["HateMM", "Multihateclip"])
    parser.add_argument("--language", default="English")
    args = parser.parse_args()

    if args.dataset_name == "HateMM":
        data_path = "./datasets/HateMM/appraise_v13_data.json"
        out_dir = "./embeddings/HateMM"
    else:
        data_path = f"./datasets/Multihateclip/{args.language}/appraise_v13_data.json"
        out_dir = f"./embeddings/Multihateclip/{args.language}"

    os.makedirs(out_dir, exist_ok=True)

    with open(data_path) as f:
        data = json.load(f)

    print(f"Loaded {len(data)} videos from {data_path}")

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModel.from_pretrained("bert-base-uncased").to(device)
    model.eval()

    # --- Prepare all text fields ---
    perception_texts = {}
    cognition_texts = {}
    answer_texts = {}
    struct_features = {}
    # Per-field answer tags
    field_names = ["which", "what", "target", "where", "why", "how"]
    field_texts = {f: {} for f in field_names}

    for item in data:
        vid = item['Video_ID']
        resp = item.get('v13_response', {})

        s1 = resp.get('step1', '')
        s2 = resp.get('step2', '')
        perception_texts[vid] = f"{s1} {s2}".strip()

        s3 = resp.get('step3', '')
        s4 = resp.get('step4', '')
        cognition_texts[vid] = f"{s3} {s4}".strip()

        answer_texts[vid] = resp.get('answer', '')

        which = resp.get('which', 'unknown').lower()
        is_hateful = 1.0 if 'hate' in which and 'non' not in which else 0.0
        struct_features[vid] = torch.tensor([is_hateful])

        for f in field_names:
            field_texts[f][vid] = resp.get(f, '')

    # --- Encode Group A ---
    print("Encoding perception (step1+step2)...")
    perception_feats = encode_texts(perception_texts, tokenizer, model, max_length=256)
    print("Encoding cognition (step3+step4)...")
    cognition_feats = encode_texts(cognition_texts, tokenizer, model, max_length=256)
    print("Encoding answer (full)...")
    answer_feats = encode_texts(answer_texts, tokenizer, model, max_length=256)

    # --- Encode Group B (per-field) ---
    field_feats = {}
    for f in field_names:
        print(f"Encoding ans_{f}...")
        field_feats[f] = encode_texts(field_texts[f], tokenizer, model, max_length=128)

    # --- Save ---
    torch.save(perception_feats, os.path.join(out_dir, "v13_perception_features.pth"))
    torch.save(cognition_feats, os.path.join(out_dir, "v13_cognition_features.pth"))
    torch.save(answer_feats, os.path.join(out_dir, "v13_answer_features.pth"))
    torch.save(struct_features, os.path.join(out_dir, "v13_struct_features.pth"))
    for f in field_names:
        torch.save(field_feats[f], os.path.join(out_dir, f"v13_ans_{f}_features.pth"))

    print(f"\nSaved v13 embeddings to {out_dir}/")
    print(f"  Group A: perception({len(perception_feats)}), cognition({len(cognition_feats)}), answer({len(answer_feats)})")
    print(f"  Group B: " + ", ".join(f"ans_{f}({len(field_feats[f])})" for f in field_names))
    print(f"  struct({len(struct_features)})")

if __name__ == "__main__":
    main()
