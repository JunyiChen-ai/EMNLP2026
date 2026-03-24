"""Ablation: Dual T1 view (v10 implicit_meaning + HVGuard Mix_description) + threshold tuning.

Also tests: 3-class training collapsed to binary, text-centric modality ablation.
"""
import argparse, csv, json, os, random, copy
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset
import warnings; warnings.filterwarnings("ignore")

device = "cuda"


class FlexDataset(Dataset):
    """Flexible dataset supporting variable modality keys."""
    def __init__(self, video_ids, features, label_map, mod_keys):
        self.video_ids = video_ids
        self.features = features
        self.label_map = label_map
        self.mod_keys = mod_keys
    def __len__(self): return len(self.video_ids)
    def __getitem__(self, idx):
        vid = self.video_ids[idx]
        out = {k: self.features[k][vid] for k in self.mod_keys}
        out["scores"] = self.features["scores"][vid]
        out["struct"] = self.features["struct"][vid]
        out["label"] = torch.tensor(self.label_map[self.features["labels"][vid]["Label"]], dtype=torch.long)
        return out

def collate_fn(batch):
    return {k: torch.stack([b[k] for b in batch]) for k in batch[0]}


class FlexACMHGF(nn.Module):
    """AC-MHGF with flexible number of modalities."""
    def __init__(self, num_mod, input_dims, num_classes=2, num_heads=4, hidden=192, score_dim=7, struct_dim=38, drop=0.15, mod_drop=0.15):
        super().__init__()
        self.projs = nn.ModuleList([nn.Sequential(nn.Linear(d, hidden), nn.GELU(), nn.Dropout(drop), nn.LayerNorm(hidden)) for d in input_dims])
        self.se = nn.Sequential(nn.Linear(score_dim, 64), nn.GELU(), nn.Linear(64, 64))
        self.struct_enc = nn.Sequential(nn.Linear(struct_dim, 64), nn.GELU(), nn.LayerNorm(64))
        self.film = nn.Linear(64, num_mod * hidden * 2)
        self.routes = nn.ModuleList([nn.Sequential(nn.Linear(hidden + 64, 128), nn.GELU(), nn.Linear(128, 1)) for _ in range(num_heads)])
        cd = num_heads * hidden + hidden + 64 + 64
        self.cls = nn.Sequential(nn.LayerNorm(cd), nn.Linear(cd, 256), nn.GELU(), nn.Dropout(drop), nn.Linear(256, 64), nn.GELU(), nn.Dropout(drop * 0.5), nn.Linear(64, num_classes))
        self.nm = num_mod; self.nh = num_heads; self.h = hidden; self.md = mod_drop

    def forward(self, batch, mod_keys, training=False):
        se = self.se(batch["scores"])
        fp = self.film(se).view(-1, self.nm, self.h, 2)
        refined = []
        for i, (proj, k) in enumerate(zip(self.projs, mod_keys)):
            h = proj(batch[k]); g, b = fp[:, i, :, 0], fp[:, i, :, 1]
            h = h * (1 + 0.1 * torch.tanh(g)) + 0.1 * b
            if training and self.md > 0:
                h = h * (torch.rand(h.size(0), 1, device=h.device) > self.md).float()
            refined.append(h)
        st = torch.stack(refined, dim=1)
        se_e = se.unsqueeze(1).expand(-1, self.nm, -1)
        ri = torch.cat([st, se_e], dim=-1)
        heads = [((st * torch.softmax(rm(ri).squeeze(-1), dim=1).unsqueeze(-1)).sum(dim=1)) for rm in self.routes]
        v_struct = self.struct_enc(batch["struct"])
        return self.cls(torch.cat(heads + [st.mean(dim=1), se, v_struct], dim=-1))


def cosine_warmup(opt, ws, ts):
    def f(s):
        if s < ws: return s / max(1, ws)
        return max(0, 0.5 * (1 + np.cos(np.pi * (s - ws) / max(1, ts - ws))))
    return LambdaLR(opt, f)


def find_best_threshold(logits_list, labels, num_classes):
    """Find threshold that maximizes accuracy (for 2-class only)."""
    if num_classes > 2:
        return None
    logits = np.array(logits_list)
    diffs = logits[:, 1] - logits[:, 0]
    best_acc, best_t = 0, 0
    for t in np.arange(-3, 3, 0.02):
        preds = (diffs > t).astype(int)
        acc = accuracy_score(labels, preds)
        if acc > best_acc: best_acc, best_t = acc, t
    return best_t


def run_config(name, features, split_ids, label_map, mod_keys, num_classes, struct_dim, num_runs=20):
    print(f"\n=== {name} ===")
    input_dims = [features[k][list(features[k].keys())[0]].shape[0] for k in mod_keys]
    score_dim = features["scores"][list(features["scores"].keys())[0]].shape[0]

    all_accs, all_accs_tuned = [], []
    for ri in range(num_runs):
        seed = ri * 1000 + 42; torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
        train_ds = FlexDataset(split_ids["train"], features, label_map, mod_keys)
        val_ds = FlexDataset(split_ids["valid"], features, label_map, mod_keys)
        test_ds = FlexDataset(split_ids["test"], features, label_map, mod_keys)
        trl = DataLoader(train_ds, 32, True, collate_fn=collate_fn)
        vl = DataLoader(val_ds, 64, False, collate_fn=collate_fn)
        tel = DataLoader(test_ds, 64, False, collate_fn=collate_fn)

        model = FlexACMHGF(len(mod_keys), input_dims, num_classes, score_dim=score_dim, struct_dim=struct_dim).to(device)
        ema = copy.deepcopy(model)
        opt = optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.02)
        ts = 45 * len(trl); ws = 5 * len(trl); sch = cosine_warmup(opt, ws, ts)
        crit = nn.CrossEntropyLoss(label_smoothing=0.03)
        bva, bst = -1, None
        for ep in range(45):
            model.train()
            for batch in trl:
                batch = {k: v.to(device) for k, v in batch.items()}; opt.zero_grad()
                crit(model(batch, mod_keys, training=True), batch["label"]).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sch.step()
                with torch.no_grad():
                    for p, ep2 in zip(model.parameters(), ema.parameters()): ep2.data.mul_(0.999).add_(p.data, alpha=0.001)
            ema.eval(); ps, ls2 = [], []
            with torch.no_grad():
                for batch in vl:
                    batch = {k: v.to(device) for k, v in batch.items()}
                    ps.extend(ema(batch, mod_keys).argmax(1).cpu().numpy()); ls2.extend(batch["label"].cpu().numpy())
            va = accuracy_score(ls2, ps)
            if va > bva: bva = va; bst = {k: v.clone() for k, v in ema.state_dict().items()}

        ema.load_state_dict(bst); ema.eval()

        # Standard eval
        ps, ls2, all_logits = [], [], []
        with torch.no_grad():
            for batch in tel:
                batch = {k: v.to(device) for k, v in batch.items()}
                logits = ema(batch, mod_keys)
                ps.extend(logits.argmax(1).cpu().numpy()); ls2.extend(batch["label"].cpu().numpy())
                all_logits.extend(logits.cpu().numpy())

        acc = accuracy_score(ls2, ps)

        # 3-class → binary collapse
        if num_classes == 3:
            binary_preds = [1 if p > 0 else 0 for p in ps]
            binary_labels = [1 if l > 0 else 0 for l in ls2]
            acc = accuracy_score(binary_labels, binary_preds)

        all_accs.append(acc)

        # Threshold tuning on val (2-class only)
        if num_classes == 2:
            val_logits, val_labs = [], []
            with torch.no_grad():
                for batch in vl:
                    batch = {k: v.to(device) for k, v in batch.items()}
                    val_logits.extend(ema(batch, mod_keys).cpu().numpy()); val_labs.extend(batch["label"].cpu().numpy())
            best_t = find_best_threshold(val_logits, val_labs, 2)
            if best_t is not None:
                test_diffs = np.array(all_logits)[:, 1] - np.array(all_logits)[:, 0]
                tuned_preds = (test_diffs > best_t).astype(int)
                tuned_acc = accuracy_score(ls2, tuned_preds)
                all_accs_tuned.append(tuned_acc)

    print(f"  Acc: {np.mean(all_accs):.4f} ± {np.std(all_accs):.4f} (max={np.max(all_accs):.4f})")
    if all_accs_tuned:
        print(f"  Acc (tuned): {np.mean(all_accs_tuned):.4f} ± {np.std(all_accs_tuned):.4f} (max={np.max(all_accs_tuned):.4f})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", required=True, choices=["English", "Chinese"])
    parser.add_argument("--num_runs", type=int, default=20)
    args = parser.parse_args()

    lang = args.language
    emb_dir = f"./embeddings/Multihateclip/{lang}"

    # Load all features
    features = {
        "text": torch.load(f"{emb_dir}/text_features.pth", map_location="cpu"),
        "audio": torch.load(f"{emb_dir}/wavlm_audio_features.pth", map_location="cpu"),
        "frame": torch.load(f"{emb_dir}/frame_features.pth", map_location="cpu"),
        "t1_v10": torch.load(f"{emb_dir}/v10_t1_features.pth", map_location="cpu"),
        "t2_v10": torch.load(f"{emb_dir}/v10_t2_features.pth", map_location="cpu"),
        "t1_hv": torch.load(f"{emb_dir}/hvguard_t1_features.pth", map_location="cpu"),
        "scores": torch.load(f"{emb_dir}/v10_scores.pth", map_location="cpu"),
        "struct": torch.load(f"{emb_dir}/v10_struct_features.pth", map_location="cpu"),
    }
    with open(f"./datasets/Multihateclip/{lang}/annotation(new).json") as f:
        features["labels"] = {d["Video_ID"]: d for d in json.load(f)}

    split_ids = {}
    for s in ["train", "valid", "test"]:
        with open(f"./datasets/Multihateclip/{lang}/splits/{s}.csv") as f:
            split_ids[s] = [r[0] for r in csv.reader(f) if r]

    # Common IDs
    feat_keys = [k for k in features if k != "labels"]
    common = set.intersection(*[set(features[k].keys()) for k in feat_keys]) & set(features["labels"].keys())
    for s in split_ids: split_ids[s] = [v for v in split_ids[s] if v in common]

    struct_dim = features["struct"][list(features["struct"].keys())[0]].shape[0]
    print(f"{lang}: {len(common)} common, train={len(split_ids['train'])}, val={len(split_ids['valid'])}, test={len(split_ids['test'])}")

    label_map_2 = {"Normal": 0, "Offensive": 1, "Hateful": 1}
    label_map_3 = {"Normal": 0, "Offensive": 1, "Hateful": 2}

    # Experiment 1: v10 only (baseline)
    run_config("v10 only (5 mod)", features, split_ids, label_map_2,
               ["text", "audio", "frame", "t1_v10", "t2_v10"], 2, struct_dim, args.num_runs)

    # Experiment 2: Dual T1 (v10 + HVGuard)
    run_config("Dual T1: v10+HV (6 mod)", features, split_ids, label_map_2,
               ["text", "audio", "frame", "t1_v10", "t2_v10", "t1_hv"], 2, struct_dim, args.num_runs)

    # Experiment 3: HVGuard T1 only (for reference)
    run_config("HVGuard T1 only (5 mod)", features, split_ids, label_map_2,
               ["text", "audio", "frame", "t1_hv", "t2_v10"], 2, struct_dim, args.num_runs)

    # Experiment 4: Text-centric (no audio/frame)
    run_config("Text-only (3 mod)", features, split_ids, label_map_2,
               ["text", "t1_v10", "t2_v10"], 2, struct_dim, args.num_runs)

    # Experiment 5: 3-class → binary collapse
    run_config("3-class collapsed", features, split_ids, label_map_3,
               ["text", "audio", "frame", "t1_v10", "t2_v10"], 3, struct_dim, args.num_runs)

    # Experiment 6: Dual T1 + 3-class collapse
    run_config("Dual T1 + 3-class", features, split_ids, label_map_3,
               ["text", "audio", "frame", "t1_v10", "t2_v10", "t1_hv"], 3, struct_dim, args.num_runs)


if __name__ == "__main__":
    main()
