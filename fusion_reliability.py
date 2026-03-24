"""Simple Reliability Gating: suppress noisy audio/frame based on text-side evidence.

Keep the simple routing-head fusion from AC-MHGF-NoScores but add:
1. Per-modality reliability gates predicted from text features
2. Modality dropout (uniform 0.15)

This is the minimal MIDAS-inspired (NeurIPS 2025) intervention.
"""
import argparse, csv, json, os, random, copy, logging
from datetime import datetime
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset
import warnings; warnings.filterwarnings("ignore")

device = "cuda"
MODALITY_KEYS = ["text", "audio", "frame", "t1", "t2"]

def setup_logger():
    os.makedirs("./logs", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    lf = f"./logs/fusion_rel_{ts}.log"
    logger = logging.getLogger("rel"); logger.setLevel(logging.INFO); logger.handlers.clear()
    fh = logging.FileHandler(lf, encoding="utf-8"); ch = logging.StreamHandler()
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    fh.setFormatter(fmt); ch.setFormatter(fmt); logger.addHandler(fh); logger.addHandler(ch)
    logger.info(f"Log: {lf}"); return logger

class DS(Dataset):
    def __init__(self, video_ids, features, label_map):
        self.video_ids = video_ids; self.f = features; self.lm = label_map
    def __len__(self): return len(self.video_ids)
    def __getitem__(self, idx):
        vid = self.video_ids[idx]
        return {k: self.f[k][vid] for k in MODALITY_KEYS} | {
            "struct": self.f["struct"][vid],
            "label": torch.tensor(self.lm[self.f["labels"][vid]["Label"]], dtype=torch.long)}
def collate_fn(batch):
    return {k: torch.stack([b[k] for b in batch]) for k in batch[0]}


class ReliabilityFusion(nn.Module):
    def __init__(self, num_mod=5, num_heads=4, hidden=192, struct_dim=9, num_classes=2, dropout=0.15, mod_drop=0.15):
        super().__init__()
        self.nm = num_mod; self.nh = num_heads; self.h = hidden; self.md = mod_drop
        self.projs = nn.ModuleList([
            nn.Sequential(nn.Linear(768, hidden), nn.GELU(), nn.Dropout(dropout), nn.LayerNorm(hidden))
            for _ in range(num_mod)])
        self.struct_enc = nn.Sequential(nn.Linear(struct_dim, 64), nn.GELU(), nn.LayerNorm(64))
        # Reliability gates for audio(1) and frame(2) — predicted from text features
        self.audio_rel = nn.Sequential(nn.Linear(hidden * 3, 64), nn.ReLU(), nn.Linear(64, 1), nn.Sigmoid())
        self.frame_rel = nn.Sequential(nn.Linear(hidden * 3, 64), nn.ReLU(), nn.Linear(64, 1), nn.Sigmoid())
        # Routing heads
        self.routes = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden, 128), nn.GELU(), nn.Linear(128, 1))
            for _ in range(num_heads)])
        cd = num_heads * hidden + hidden + 64
        self.cls = nn.Sequential(
            nn.LayerNorm(cd), nn.Linear(cd, 256), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(256, 64), nn.GELU(), nn.Dropout(dropout * 0.5),
            nn.Linear(64, num_classes))

    def forward(self, batch, training=False):
        refined = []
        for i, (proj, k) in enumerate(zip(self.projs, MODALITY_KEYS)):
            h = proj(batch[k])
            if training and self.md > 0:
                h = h * (torch.rand(h.size(0), 1, device=h.device) > self.md).float()
            refined.append(h)

        # Text context for reliability prediction
        text_ctx = torch.cat([refined[0], refined[3], refined[4]], dim=-1)  # text + T1 + T2

        # Reliability gating on audio(1) and frame(2)
        ar = self.audio_rel(text_ctx)
        fr = self.frame_rel(text_ctx)
        refined[1] = refined[1] * ar  # gate audio
        refined[2] = refined[2] * fr  # gate frame

        st = torch.stack(refined, dim=1)
        heads = []
        for rm in self.routes:
            w = torch.softmax(rm(st).squeeze(-1), dim=1)
            heads.append((st * w.unsqueeze(-1)).sum(dim=1))
        v_struct = self.struct_enc(batch["struct"])
        return self.cls(torch.cat(heads + [st.mean(dim=1), v_struct], dim=-1))


def cosine_warmup(opt, ws, ts):
    def f(s):
        if s < ws: return s / max(1, ws)
        return max(0, 0.5 * (1 + np.cos(np.pi * (s - ws) / max(1, ts - ws))))
    return LambdaLR(opt, f)

def load_split_ids(split_dir):
    s = {}
    for n in ["train", "valid", "test"]:
        with open(os.path.join(split_dir, f"{n}.csv")) as f:
            s[n] = [r[0] for r in csv.reader(f) if r]
    return s

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", default="HateMM", choices=["HateMM", "Multihateclip"])
    parser.add_argument("--language", default="English")
    parser.add_argument("--num_runs", type=int, default=20)
    args = parser.parse_args()
    logger = setup_logger()

    if args.dataset_name == "HateMM":
        emb_dir = "./embeddings/HateMM"; ann_path = "./datasets/HateMM/annotation(new).json"
        split_dir = "./datasets/HateMM/splits"; label_map = {"Non Hate": 0, "Hate": 1}
    else:
        emb_dir = f"./embeddings/Multihateclip/{args.language}"
        ann_path = f"./datasets/Multihateclip/{args.language}/annotation(new).json"
        split_dir = f"./datasets/Multihateclip/{args.language}/splits"
        label_map = {"Normal": 0, "Offensive": 1, "Hateful": 1}

    features = {
        "text": torch.load(f"{emb_dir}/text_features.pth", map_location="cpu"),
        "audio": torch.load(f"{emb_dir}/wavlm_audio_features.pth", map_location="cpu"),
        "frame": torch.load(f"{emb_dir}/frame_features.pth", map_location="cpu"),
        "t1": torch.load(f"{emb_dir}/v12_t1_features.pth", map_location="cpu"),
        "t2": torch.load(f"{emb_dir}/v12_t2_features.pth", map_location="cpu"),
        "struct": torch.load(f"{emb_dir}/v12_struct_features.pth", map_location="cpu"),
    }
    with open(ann_path) as f: features["labels"] = {d["Video_ID"]: d for d in json.load(f)}
    split_ids = load_split_ids(split_dir)
    feat_keys = [k for k in features if k != "labels"]
    common = set.intersection(*[set(features[k].keys()) for k in feat_keys]) & set(features["labels"].keys())
    for s in split_ids: split_ids[s] = [v for v in split_ids[s] if v in common]
    struct_dim = features["struct"][list(features["struct"].keys())[0]].shape[0]
    logger.info(f"{args.dataset_name} {args.language}: common={len(common)}, struct_dim={struct_dim}")

    accs, mf1s = [], []
    for ri in range(args.num_runs):
        seed = ri * 1000 + 42; torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
        trd = DS(split_ids["train"], features, label_map); vd = DS(split_ids["valid"], features, label_map); ted = DS(split_ids["test"], features, label_map)
        trl = DataLoader(trd, 32, True, collate_fn=collate_fn); vl = DataLoader(vd, 64, False, collate_fn=collate_fn); tel = DataLoader(ted, 64, False, collate_fn=collate_fn)
        model = ReliabilityFusion(struct_dim=struct_dim).to(device); ema = copy.deepcopy(model)
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
        ema.load_state_dict(bst); ema.eval(); ps, ls2 = [], []
        with torch.no_grad():
            for batch in tel:
                batch = {k: v.to(device) for k, v in batch.items()}
                ps.extend(ema(batch).argmax(1).cpu().numpy()); ls2.extend(batch["label"].cpu().numpy())
        acc = accuracy_score(ls2, ps); mf1 = f1_score(ls2, ps, average="macro", zero_division=0)
        accs.append(acc); mf1s.append(mf1)
        mk = " ***" if acc >= 0.90 else (" **" if acc >= 0.85 else "")
        logger.info(f"  Run {ri+1}/{args.num_runs}: Acc={acc:.4f} M-F1={mf1:.4f}{mk}")
    logger.info(f"  SUMMARY: Acc={np.mean(accs):.4f}±{np.std(accs):.4f} (max={np.max(accs):.4f})")
    logger.info(f"  >=0.85: {sum(1 for a in accs if a>=0.85)}, >=0.88: {sum(1 for a in accs if a>=0.88)}, >=0.90: {sum(1 for a in accs if a>=0.90)}")

if __name__ == "__main__":
    main()
