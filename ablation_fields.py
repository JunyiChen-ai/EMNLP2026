"""Ablation: Systematically remove each of the 4 LLM output fields on all datasets.

Fields:
- T1 (implicit_meaning) → 768d BERT embedding
- T2 (contrastive_readings) → 768d BERT embedding
- scores (appraisal_vector) → 7d → FiLM conditioning
- struct (stance + multi-sample stats) → 36d

Also tests removing audio and frame modalities.

Uses the same AC-MHGF architecture from main.py but with flexible modality configs.
"""
import argparse, csv, json, os, random, copy
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, Dataset
import warnings; warnings.filterwarnings("ignore")

device = "cuda"


class AblationDataset(Dataset):
    def __init__(self, video_ids, features, label_map, use_keys):
        self.video_ids = video_ids
        self.features = features
        self.label_map = label_map
        self.use_keys = use_keys

    def __len__(self):
        return len(self.video_ids)

    def __getitem__(self, idx):
        vid = self.video_ids[idx]
        out = {k: self.features[k][vid] for k in self.use_keys if k in self.features}
        out["label"] = torch.tensor(self.label_map[self.features["labels"][vid]["Label"]], dtype=torch.long)
        return out


def collate_fn(batch):
    return {k: torch.stack([b[k] for b in batch]) for k in batch[0]}


class AblationACMHGF(nn.Module):
    """AC-MHGF that handles missing modalities/fields gracefully."""
    def __init__(self, mod_keys, mod_dims, has_scores=True, score_dim=7,
                 has_struct=True, struct_dim=36, num_classes=2,
                 num_heads=4, hidden=192, dropout=0.15, mod_drop=0.15):
        super().__init__()
        self.mod_keys = mod_keys
        self.nm = len(mod_keys)
        self.has_scores = has_scores
        self.has_struct = has_struct
        self.h = hidden
        self.md = mod_drop

        self.projs = nn.ModuleList([
            nn.Sequential(nn.Linear(d, hidden), nn.GELU(), nn.Dropout(dropout), nn.LayerNorm(hidden))
            for d in mod_dims
        ])

        if has_scores:
            self.se = nn.Sequential(nn.Linear(score_dim, 64), nn.GELU(), nn.Linear(64, 64))
            self.film = nn.Linear(64, self.nm * hidden * 2)
        else:
            self.se = nn.Sequential(nn.Linear(1, 64), nn.GELU(), nn.Linear(64, 64))  # dummy
            self.film = nn.Linear(64, self.nm * hidden * 2)

        if has_struct:
            self.struct_enc = nn.Sequential(nn.Linear(struct_dim, 64), nn.GELU(), nn.LayerNorm(64))
        else:
            self.struct_enc = None

        self.routes = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden + 64, 128), nn.GELU(), nn.Linear(128, 1))
            for _ in range(num_heads)
        ])

        cd = num_heads * hidden + hidden + 64
        if has_struct:
            cd += 64
        self.cls = nn.Sequential(
            nn.LayerNorm(cd), nn.Linear(cd, 256), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(256, 64), nn.GELU(), nn.Dropout(dropout * 0.5),
            nn.Linear(64, num_classes)
        )
        self.nh = num_heads

    def forward(self, batch, training=False):
        if self.has_scores:
            se = self.se(batch["scores"])
        else:
            se = self.se(torch.zeros(batch[self.mod_keys[0]].size(0), 1, device=batch[self.mod_keys[0]].device))

        fp = self.film(se).view(-1, self.nm, self.h, 2)
        refined = []
        for i, (proj, k) in enumerate(zip(self.projs, self.mod_keys)):
            h = proj(batch[k])
            g, b = fp[:, i, :, 0], fp[:, i, :, 1]
            h = h * (1 + 0.1 * torch.tanh(g)) + 0.1 * b
            if training and self.md > 0:
                h = h * (torch.rand(h.size(0), 1, device=h.device) > self.md).float()
            refined.append(h)

        st = torch.stack(refined, dim=1)
        se_e = se.unsqueeze(1).expand(-1, self.nm, -1)
        ri = torch.cat([st, se_e], dim=-1)
        heads = []
        for rm in self.routes:
            w = torch.softmax(rm(ri).squeeze(-1), dim=1)
            heads.append((st * w.unsqueeze(-1)).sum(dim=1))

        parts = heads + [st.mean(dim=1), se]
        if self.has_struct and "struct" in batch:
            parts.append(self.struct_enc(batch["struct"]))

        return self.cls(torch.cat(parts, dim=-1))


def cosine_warmup(opt, ws, ts):
    def f(s):
        if s < ws: return s / max(1, ws)
        return max(0, 0.5 * (1 + np.cos(np.pi * (s - ws) / max(1, ts - ws))))
    return LambdaLR(opt, f)


def run_ablation(name, features, split_ids, label_map, mod_keys, has_scores, has_struct,
                 num_classes=2, num_runs=20):
    # Get dims
    vid0 = list(features[mod_keys[0]].keys())[0]
    mod_dims = [features[k][vid0].shape[0] for k in mod_keys]
    score_dim = features["scores"][vid0].shape[0] if has_scores else 1
    struct_dim = features["struct"][vid0].shape[0] if has_struct else 1

    use_keys = list(mod_keys)
    if has_scores: use_keys.append("scores")
    if has_struct: use_keys.append("struct")

    feat_keys = use_keys
    common = set.intersection(*[set(features[k].keys()) for k in feat_keys]) & set(features["labels"].keys())
    cur = {s: [v for v in split_ids[s] if v in common] for s in split_ids}

    print(f"\n=== {name} === (mods={mod_keys}, scores={has_scores}, struct={has_struct}, common={len(common)}, test={len(cur['test'])})")

    accs = []
    for ri in range(num_runs):
        seed = ri * 1000 + 42; torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
        trd = AblationDataset(cur["train"], features, label_map, use_keys)
        vd = AblationDataset(cur["valid"], features, label_map, use_keys)
        ted = AblationDataset(cur["test"], features, label_map, use_keys)
        trl = DataLoader(trd, 32, True, collate_fn=collate_fn)
        vl = DataLoader(vd, 64, False, collate_fn=collate_fn)
        tel = DataLoader(ted, 64, False, collate_fn=collate_fn)

        model = AblationACMHGF(
            mod_keys, mod_dims, has_scores, score_dim,
            has_struct, struct_dim, num_classes
        ).to(device)
        ema = copy.deepcopy(model)
        opt = optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.02)
        ts = 45 * len(trl); ws = 5 * len(trl); sch = cosine_warmup(opt, ws, ts)
        crit = nn.CrossEntropyLoss(label_smoothing=0.03)
        bva, bst = -1, None

        for ep in range(45):
            model.train()
            for batch in trl:
                batch = {k: v.to(device) for k, v in batch.items()}
                opt.zero_grad()
                crit(model(batch, training=True), batch["label"]).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); sch.step()
                with torch.no_grad():
                    for p, ep2 in zip(model.parameters(), ema.parameters()):
                        ep2.data.mul_(0.999).add_(p.data, alpha=0.001)

            ema.eval(); ps, ls2 = [], []
            with torch.no_grad():
                for batch in vl:
                    batch = {k: v.to(device) for k, v in batch.items()}
                    ps.extend(ema(batch).argmax(1).cpu().numpy())
                    ls2.extend(batch["label"].cpu().numpy())
            va = accuracy_score(ls2, ps)
            if va > bva: bva = va; bst = {k: v.clone() for k, v in ema.state_dict().items()}

        ema.load_state_dict(bst); ema.eval(); ps, ls2 = [], []
        with torch.no_grad():
            for batch in tel:
                batch = {k: v.to(device) for k, v in batch.items()}
                ps.extend(ema(batch).argmax(1).cpu().numpy())
                ls2.extend(batch["label"].cpu().numpy())
        accs.append(accuracy_score(ls2, ps))

    print(f"  Acc: {np.mean(accs):.4f} ± {np.std(accs):.4f} (max={np.max(accs):.4f})")
    return np.mean(accs), np.max(accs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", default="HateMM", choices=["HateMM", "Multihateclip"])
    parser.add_argument("--language", default="English", choices=["English", "Chinese"])
    parser.add_argument("--num_runs", type=int, default=10)
    args = parser.parse_args()

    if args.dataset_name == "HateMM":
        emb_dir = "./embeddings/HateMM"
        ann_path = "./datasets/HateMM/annotation(new).json"
        split_dir = "./datasets/HateMM/splits"
        label_map = {"Non Hate": 0, "Hate": 1}
    else:
        emb_dir = f"./embeddings/Multihateclip/{args.language}"
        ann_path = f"./datasets/Multihateclip/{args.language}/annotation(new).json"
        split_dir = f"./datasets/Multihateclip/{args.language}/splits"
        label_map = {"Normal": 0, "Offensive": 1, "Hateful": 1}

    features = {
        "text": torch.load(f"{emb_dir}/text_features.pth", map_location="cpu"),
        "audio": torch.load(f"{emb_dir}/wavlm_audio_features.pth", map_location="cpu"),
        "frame": torch.load(f"{emb_dir}/frame_features.pth", map_location="cpu"),
        "t1": torch.load(f"{emb_dir}/v9_t1_features.pth", map_location="cpu"),
        "t2": torch.load(f"{emb_dir}/v9_t2_features.pth", map_location="cpu"),
        "scores": torch.load(f"{emb_dir}/v9_scores.pth", map_location="cpu"),
        "struct": torch.load(f"{emb_dir}/v9_struct_features.pth", map_location="cpu"),
    }
    with open(ann_path) as f:
        features["labels"] = {d["Video_ID"]: d for d in json.load(f)}

    split_ids = {}
    for s in ["train", "valid", "test"]:
        with open(os.path.join(split_dir, f"{s}.csv")) as f:
            split_ids[s] = [r[0] for r in csv.reader(f) if r]

    nc = 2
    nr = args.num_runs

    print(f"Dataset: {args.dataset_name} {args.language}")
    print("=" * 60)

    # Full model
    run_ablation("Full (text+audio+frame+T1+T2, scores, struct)",
                 features, split_ids, label_map,
                 ["text", "audio", "frame", "t1", "t2"], True, True, nc, nr)

    # Remove T1
    run_ablation("Remove T1 (implicit_meaning)",
                 features, split_ids, label_map,
                 ["text", "audio", "frame", "t2"], True, True, nc, nr)

    # Remove T2
    run_ablation("Remove T2 (contrastive_readings)",
                 features, split_ids, label_map,
                 ["text", "audio", "frame", "t1"], True, True, nc, nr)

    # Remove scores (no FiLM conditioning)
    run_ablation("Remove scores (no FiLM)",
                 features, split_ids, label_map,
                 ["text", "audio", "frame", "t1", "t2"], False, True, nc, nr)

    # Remove struct (no stance/stats)
    run_ablation("Remove struct (no stance/stats)",
                 features, split_ids, label_map,
                 ["text", "audio", "frame", "t1", "t2"], True, False, nc, nr)

    # Remove audio
    run_ablation("Remove audio",
                 features, split_ids, label_map,
                 ["text", "frame", "t1", "t2"], True, True, nc, nr)

    # Remove frame
    run_ablation("Remove frame",
                 features, split_ids, label_map,
                 ["text", "audio", "t1", "t2"], True, True, nc, nr)

    # Remove both audio + frame (text-only)
    run_ablation("Text-only (no audio, no frame)",
                 features, split_ids, label_map,
                 ["text", "t1", "t2"], True, True, nc, nr)

    # Remove T1 + T2 (no LLM text)
    run_ablation("No LLM text (only text+audio+frame+scores+struct)",
                 features, split_ids, label_map,
                 ["text", "audio", "frame"], True, True, nc, nr)

    # Only T1 + scores (minimal LLM)
    run_ablation("Only T1+scores (minimal)",
                 features, split_ids, label_map,
                 ["t1"], True, True, nc, nr)


if __name__ == "__main__":
    main()
