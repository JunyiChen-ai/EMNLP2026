"""Ablation runner: test different T1 sources while keeping everything else the same.

Usage:
    python ablation_run.py --language English --t1_source ours      # our v9 implicit_meaning
    python ablation_run.py --language English --t1_source hvguard   # HVGuard's Mix_description
"""
import argparse, csv, json, os, random, copy
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset
import warnings; warnings.filterwarnings("ignore")

device = "cuda"
MODALITY_KEYS = ["text", "audio", "frame", "t1", "t2"]


class HateMM_Dataset(Dataset):
    def __init__(self, video_ids, features, label_map):
        self.video_ids = video_ids
        self.features = features
        self.label_map = label_map
    def __len__(self): return len(self.video_ids)
    def __getitem__(self, idx):
        vid = self.video_ids[idx]
        return {
            "text": self.features["text"][vid], "audio": self.features["audio"][vid],
            "frame": self.features["frame"][vid], "t1": self.features["t1"][vid],
            "t2": self.features["t2"][vid], "scores": self.features["scores"][vid],
            "struct": self.features["struct"][vid],
            "label": torch.tensor(self.label_map[self.features["labels"][vid]["Label"]], dtype=torch.long),
        }

def collate_fn(batch):
    return {k: torch.stack([b[k] for b in batch]) for k in batch[0]}


# Import AC_MHGF from main.py
from main import AC_MHGF, get_cosine_schedule_with_warmup, load_split_ids


def run(args):
    lang = args.language
    emb_dir = f"./embeddings/Multihateclip/{lang}"
    ann_path = f"./datasets/Multihateclip/{lang}/annotation(new).json"
    split_dir = f"./datasets/Multihateclip/{lang}/splits"

    if args.num_classes == 2:
        label_map = {"Normal": 0, "Offensive": 1, "Hateful": 1}
    else:
        label_map = {"Normal": 0, "Offensive": 1, "Hateful": 2}

    # Select T1 source
    if args.t1_source == "ours":
        t1_path = f"{emb_dir}/v9_t1_features.pth"
    else:
        t1_path = f"{emb_dir}/hvguard_t1_features.pth"

    features = {
        "text": torch.load(f"{emb_dir}/text_features.pth", map_location="cpu"),
        "audio": torch.load(f"{emb_dir}/wavlm_audio_features.pth", map_location="cpu"),
        "frame": torch.load(f"{emb_dir}/frame_features.pth", map_location="cpu"),
        "t1": torch.load(t1_path, map_location="cpu"),
        "t2": torch.load(f"{emb_dir}/v9_t2_features.pth", map_location="cpu"),
        "scores": torch.load(f"{emb_dir}/v9_scores.pth", map_location="cpu"),
        "struct": torch.load(f"{emb_dir}/v9_struct_features.pth", map_location="cpu"),
    }
    with open(ann_path) as f:
        features["labels"] = {item["Video_ID"]: item for item in json.load(f)}

    split_ids = load_split_ids(split_dir)
    feat_keys = [k for k in features if k != "labels"]
    common_ids = set.intersection(*[set(features[k].keys()) for k in feat_keys]) & set(features["labels"].keys())
    for s in split_ids:
        split_ids[s] = [v for v in split_ids[s] if v in common_ids]

    struct_dim = features["struct"][list(features["struct"].keys())[0]].shape[0]
    print(f"{lang} | T1={args.t1_source} | {args.num_classes}-class | Data: {len(common_ids)}, "
          f"train={len(split_ids['train'])}, val={len(split_ids['valid'])}, test={len(split_ids['test'])}")

    all_results = []
    for run_idx in range(args.num_runs):
        seed = run_idx * 1000 + 42
        torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
        train_ds = HateMM_Dataset(split_ids["train"], features, label_map)
        val_ds = HateMM_Dataset(split_ids["valid"], features, label_map)
        test_ds = HateMM_Dataset(split_ids["test"], features, label_map)
        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, collate_fn=collate_fn)
        test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, collate_fn=collate_fn)

        model = AC_MHGF(struct_dim=struct_dim, num_classes=args.num_classes, modality_dropout=0.15).to(device)
        ema = copy.deepcopy(model)
        optimizer = optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.02)
        total_steps = 45 * len(train_loader)
        warmup_steps = 5 * len(train_loader)
        scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
        criterion = nn.CrossEntropyLoss(label_smoothing=0.03)

        best_val_acc, best_state = -1, None
        for epoch in range(45):
            model.train()
            for batch in train_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                optimizer.zero_grad()
                criterion(model(batch, training=True), batch["label"]).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step(); scheduler.step()
                with torch.no_grad():
                    for p, ep in zip(model.parameters(), ema.parameters()):
                        ep.data.mul_(0.999).add_(p.data, alpha=0.001)
            ema.eval()
            preds, labs = [], []
            with torch.no_grad():
                for batch in val_loader:
                    batch = {k: v.to(device) for k, v in batch.items()}
                    preds.extend(ema(batch).argmax(1).cpu().numpy())
                    labs.extend(batch["label"].cpu().numpy())
            va = accuracy_score(labs, preds)
            if va > best_val_acc:
                best_val_acc = va
                best_state = {k: v.clone() for k, v in ema.state_dict().items()}

        ema.load_state_dict(best_state); ema.eval()
        preds, labs = [], []
        with torch.no_grad():
            for batch in test_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                preds.extend(ema(batch).argmax(1).cpu().numpy())
                labs.extend(batch["label"].cpu().numpy())
        acc = accuracy_score(labs, preds)
        mf1 = f1_score(labs, preds, average="macro", zero_division=0)
        all_results.append({"acc": acc, "m_f1": mf1})

    accs = [r["acc"] for r in all_results]
    mf1s = [r["m_f1"] for r in all_results]
    print(f"  Acc: {np.mean(accs):.4f} ± {np.std(accs):.4f} (max={np.max(accs):.4f})")
    print(f"  M-F1: {np.mean(mf1s):.4f} ± {np.std(mf1s):.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", required=True, choices=["English", "Chinese"])
    parser.add_argument("--t1_source", required=True, choices=["ours", "hvguard"])
    parser.add_argument("--num_classes", type=int, default=2, choices=[2, 3])
    parser.add_argument("--num_runs", type=int, default=20)
    args = parser.parse_args()
    run(args)
