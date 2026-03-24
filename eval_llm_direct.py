"""Evaluate LLM direct prediction accuracy (no fusion/kNN).

Compares v13 (P2C-CoT) vs baseline (direct judge) on raw LLM output.
"""
import json, argparse
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

def eval_dataset(dataset_name, language="English"):
    if dataset_name == "HateMM":
        ann_path = "./datasets/HateMM/annotation(new).json"
        v13_path = "./datasets/HateMM/appraise_v13_data.json"
        base_path = "./datasets/HateMM/baseline_direct_data.json"
        hate_label = "Hate"
    else:
        ann_path = f"./datasets/Multihateclip/{language}/annotation(new).json"
        v13_path = f"./datasets/Multihateclip/{language}/appraise_v13_data.json"
        base_path = f"./datasets/Multihateclip/{language}/baseline_direct_data.json"
        hate_label = None  # Offensive+Hateful = 1

    with open(ann_path) as f:
        ann = {d["Video_ID"]: d for d in json.load(f)}

    def true_label(vid):
        label = ann[vid]["Label"]
        if dataset_name == "HateMM":
            return 1 if label == "Hate" else 0
        else:
            return 1 if label in ("Offensive", "Hateful") else 0

    def pred_label(which):
        if not which: return None
        low = which.lower()
        if "non" in low: return 0
        if "hateful" in low or "hate" in low: return 1
        return None

    tag = dataset_name if dataset_name == "HateMM" else f"MHC_{language[:2]}"
    print(f"\n{'='*60}")
    print(f"  {tag}")
    print(f"{'='*60}")

    for name, path, resp_key in [("v13 (P2C-CoT)", v13_path, "v13_response"),
                                   ("Baseline (direct)", base_path, "baseline_response")]:
        try:
            with open(path) as f:
                data = json.load(f)
        except FileNotFoundError:
            print(f"  {name}: FILE NOT FOUND ({path})")
            continue

        y_true, y_pred = [], []
        missing = 0
        for item in data:
            vid = item.get("Video_ID")
            if vid not in ann: continue
            resp = item.get(resp_key, {})
            which = resp.get("which", "")
            p = pred_label(which)
            if p is None:
                missing += 1
                continue
            y_true.append(true_label(vid))
            y_pred.append(p)

        if not y_true:
            print(f"  {name}: no valid predictions")
            continue

        acc = accuracy_score(y_true, y_pred)
        mf1 = f1_score(y_true, y_pred, average='macro')
        mp = precision_score(y_true, y_pred, average='macro')
        mr = recall_score(y_true, y_pred, average='macro')
        cm = confusion_matrix(y_true, y_pred)

        print(f"\n  {name} ({len(y_true)} valid, {missing} unparseable)")
        print(f"    ACC={acc:.4f}  M-F1={mf1:.4f}  M-P={mp:.4f}  M-R={mr:.4f}")
        print(f"    CM: {cm.tolist()}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", default="all", choices=["HateMM", "Multihateclip", "all"])
    parser.add_argument("--language", default="English")
    args = parser.parse_args()

    if args.dataset_name == "all":
        eval_dataset("HateMM")
        eval_dataset("Multihateclip", "English")
        eval_dataset("Multihateclip", "Chinese")
    elif args.dataset_name == "Multihateclip":
        eval_dataset("Multihateclip", args.language)
    else:
        eval_dataset("HateMM")


if __name__ == "__main__":
    main()
