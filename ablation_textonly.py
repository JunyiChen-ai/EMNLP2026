"""Ablation: Text-only modalities (drop audio and frame) with threshold tuning.
Tests whether audio/frame are noise for MultiHateClip.
"""
import argparse, csv, json, os, random, copy
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset
import warnings; warnings.filterwarnings("ignore")

device = "cuda"


class TextOnlyDataset(Dataset):
    def __init__(self, video_ids, features, label_map, use_v10=True):
        self.video_ids = video_ids; self.f = features; self.lm = label_map; self.v10 = use_v10
    def __len__(self): return len(self.video_ids)
    def __getitem__(self, idx):
        vid = self.video_ids[idx]
        t1_key = "t1_v10" if self.v10 and "t1_v10" in self.f else "t1"
        t2_key = "t2_v10" if self.v10 and "t2_v10" in self.f else "t2"
        scores_key = "scores_v10" if self.v10 and "scores_v10" in self.f else "scores"
        struct_key = "struct_v10" if self.v10 and "struct_v10" in self.f else "struct"
        return {
            "text": self.f["text"][vid],
            "t1": self.f[t1_key][vid],
            "t2": self.f[t2_key][vid],
            "scores": self.f[scores_key][vid],
            "struct": self.f[struct_key][vid],
            "label": torch.tensor(self.lm[self.f["labels"][vid]["Label"]], dtype=torch.long),
        }

def collate_fn(batch):
    return {k: torch.stack([b[k] for b in batch]) for k in batch[0]}

MODS = ["text", "t1", "t2"]

class TextOnlyACMHGF(nn.Module):
    def __init__(self, num_classes=2, hidden=192, score_dim=7, struct_dim=38, drop=0.15, md=0.15):
        super().__init__()
        self.projs = nn.ModuleList([nn.Sequential(nn.Linear(768, hidden), nn.GELU(), nn.Dropout(drop), nn.LayerNorm(hidden)) for _ in range(3)])
        self.se = nn.Sequential(nn.Linear(score_dim, 64), nn.GELU(), nn.Linear(64, 64))
        self.struct_enc = nn.Sequential(nn.Linear(struct_dim, 64), nn.GELU(), nn.LayerNorm(64))
        self.film = nn.Linear(64, 3 * hidden * 2)
        self.routes = nn.ModuleList([nn.Sequential(nn.Linear(hidden+64, 128), nn.GELU(), nn.Linear(128, 1)) for _ in range(4)])
        cd = 4 * hidden + hidden + 64 + 64
        self.cls = nn.Sequential(nn.LayerNorm(cd), nn.Linear(cd, 256), nn.GELU(), nn.Dropout(drop), nn.Linear(256, 64), nn.GELU(), nn.Dropout(drop*0.5), nn.Linear(64, num_classes))
        self.h = hidden; self.md = md

    def forward(self, batch, training=False):
        se = self.se(batch["scores"]); fp = self.film(se).view(-1, 3, self.h, 2)
        refined = []
        for i, (proj, k) in enumerate(zip(self.projs, MODS)):
            h = proj(batch[k]); g, b = fp[:, i, :, 0], fp[:, i, :, 1]
            h = h * (1 + 0.1 * torch.tanh(g)) + 0.1 * b
            if training and self.md > 0: h = h * (torch.rand(h.size(0), 1, device=h.device) > self.md).float()
            refined.append(h)
        st = torch.stack(refined, dim=1)
        se_e = se.unsqueeze(1).expand(-1, 3, -1)
        ri = torch.cat([st, se_e], dim=-1)
        heads = [((st * torch.softmax(rm(ri).squeeze(-1), dim=1).unsqueeze(-1)).sum(dim=1)) for rm in self.routes]
        v_struct = self.struct_enc(batch["struct"])
        return self.cls(torch.cat(heads + [st.mean(dim=1), se, v_struct], dim=-1))

def cosine_warmup(opt, ws, ts):
    def f(s):
        if s < ws: return s / max(1, ws)
        return max(0, 0.5 * (1 + np.cos(np.pi * (s - ws) / max(1, ts - ws))))
    return LambdaLR(opt, f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", required=True)
    parser.add_argument("--prompt_version", default="v10", choices=["v9", "v10"])
    parser.add_argument("--num_runs", type=int, default=20)
    args = parser.parse_args()

    lang = args.language
    emb_dir = f"./embeddings/Multihateclip/{lang}"
    label_map = {"Normal": 0, "Offensive": 1, "Hateful": 1}

    features = {"text": torch.load(f"{emb_dir}/text_features.pth", map_location="cpu")}
    if args.prompt_version == "v10":
        features["t1"] = torch.load(f"{emb_dir}/v10_t1_features.pth", map_location="cpu")
        features["t2"] = torch.load(f"{emb_dir}/v10_t2_features.pth", map_location="cpu")
        features["scores"] = torch.load(f"{emb_dir}/v10_scores.pth", map_location="cpu")
        features["struct"] = torch.load(f"{emb_dir}/v10_struct_features.pth", map_location="cpu")
    else:
        features["t1"] = torch.load(f"{emb_dir}/v9_t1_features.pth", map_location="cpu")
        features["t2"] = torch.load(f"{emb_dir}/v9_t2_features.pth", map_location="cpu")
        features["scores"] = torch.load(f"{emb_dir}/v9_scores.pth", map_location="cpu")
        features["struct"] = torch.load(f"{emb_dir}/v9_struct_features.pth", map_location="cpu")

    with open(f"./datasets/Multihateclip/{lang}/annotation(new).json") as f:
        features["labels"] = {d["Video_ID"]: d for d in json.load(f)}

    split_ids = {}
    for s in ["train", "valid", "test"]:
        with open(f"./datasets/Multihateclip/{lang}/splits/{s}.csv") as f:
            split_ids[s] = [r[0] for r in csv.reader(f) if r]

    feat_keys = [k for k in features if k != "labels"]
    common = set.intersection(*[set(features[k].keys()) for k in feat_keys]) & set(features["labels"].keys())
    for s in split_ids: split_ids[s] = [v for v in split_ids[s] if v in common]

    struct_dim = features["struct"][list(features["struct"].keys())[0]].shape[0]
    score_dim = features["scores"][list(features["scores"].keys())[0]].shape[0]
    print(f"{lang} text-only ({args.prompt_version}): {len(common)} videos, train={len(split_ids['train'])}, val={len(split_ids['valid'])}, test={len(split_ids['test'])}")

    all_accs, all_tuned = [], []
    for ri in range(args.num_runs):
        seed = ri * 1000 + 42; torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
        trd = TextOnlyDataset(split_ids["train"], features, label_map, False)
        vd = TextOnlyDataset(split_ids["valid"], features, label_map, False)
        ted = TextOnlyDataset(split_ids["test"], features, label_map, False)
        trl = DataLoader(trd, 32, True, collate_fn=collate_fn); vl = DataLoader(vd, 64, False, collate_fn=collate_fn); tel = DataLoader(ted, 64, False, collate_fn=collate_fn)

        model = TextOnlyACMHGF(struct_dim=struct_dim, score_dim=score_dim).to(device)
        ema = copy.deepcopy(model)
        opt = optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.02)
        ts = 45 * len(trl); ws = 5 * len(trl); sch = cosine_warmup(opt, ws, ts)
        crit = nn.CrossEntropyLoss(label_smoothing=0.03); bva, bst = -1, None
        for ep in range(45):
            model.train()
            for batch in trl:
                batch = {k: v.to(device) for k, v in batch.items()}; opt.zero_grad()
                crit(model(batch, training=True), batch["label"]).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sch.step()
                with torch.no_grad():
                    for p, ep2 in zip(model.parameters(), ema.parameters()): ep2.data.mul_(0.999).add_(p.data, alpha=0.001)
            ema.eval(); ps, ls2 = [], []
            with torch.no_grad():
                for batch in vl:
                    batch = {k: v.to(device) for k, v in batch.items()}
                    ps.extend(ema(batch).argmax(1).cpu().numpy()); ls2.extend(batch["label"].cpu().numpy())
            va = accuracy_score(ls2, ps)
            if va > bva: bva = va; bst = {k: v.clone() for k, v in ema.state_dict().items()}

        ema.load_state_dict(bst); ema.eval()
        ps, ls2, logits_all = [], [], []
        with torch.no_grad():
            for batch in tel:
                batch = {k: v.to(device) for k, v in batch.items()}
                logits = ema(batch)
                ps.extend(logits.argmax(1).cpu().numpy()); ls2.extend(batch["label"].cpu().numpy())
                logits_all.extend(logits.cpu().numpy())
        acc = accuracy_score(ls2, ps); all_accs.append(acc)

        # Threshold tuning
        val_logits, val_labs = [], []
        with torch.no_grad():
            for batch in vl:
                batch = {k: v.to(device) for k, v in batch.items()}
                val_logits.extend(ema(batch).cpu().numpy()); val_labs.extend(batch["label"].cpu().numpy())
        vd2 = np.array(val_logits)[:, 1] - np.array(val_logits)[:, 0]
        td = np.array(logits_all)[:, 1] - np.array(logits_all)[:, 0]
        bt, bva2 = 0, 0
        for t in np.arange(-3, 3, 0.02):
            va2 = accuracy_score(val_labs, (vd2 > t).astype(int))
            if va2 > bva2: bva2, bt = va2, t
        tuned = accuracy_score(ls2, (td > bt).astype(int))
        all_tuned.append(tuned)

    print(f"  Acc: {np.mean(all_accs):.4f} ± {np.std(all_accs):.4f} (max={np.max(all_accs):.4f})")
    print(f"  Acc (tuned): {np.mean(all_tuned):.4f} ± {np.std(all_tuned):.4f} (max={np.max(all_tuned):.4f})")

if __name__ == "__main__":
    main()
