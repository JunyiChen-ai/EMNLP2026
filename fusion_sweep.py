"""Hyperparameter sweep on v12 AC-MHGF-NoScores fusion.

Try: different hidden sizes, lr, epochs, dropout, number of heads.
Also try: text-only (drop audio/frame) for Chinese.
Run on all 3 datasets.
"""
import argparse, csv, json, os, random, copy, logging
from datetime import datetime
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, Dataset
import warnings; warnings.filterwarnings("ignore")

device = "cuda"

def setup_logger():
    os.makedirs("./logs", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    lf = f"./logs/fusion_sweep_{ts}.log"
    logger = logging.getLogger("sweep"); logger.setLevel(logging.INFO); logger.handlers.clear()
    fh = logging.FileHandler(lf, encoding="utf-8"); ch = logging.StreamHandler()
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    fh.setFormatter(fmt); ch.setFormatter(fmt); logger.addHandler(fh); logger.addHandler(ch)
    return logger

class DS(Dataset):
    def __init__(self, video_ids, features, label_map, mod_keys):
        self.video_ids = video_ids; self.f = features; self.lm = label_map; self.mk = mod_keys
    def __len__(self): return len(self.video_ids)
    def __getitem__(self, idx):
        vid = self.video_ids[idx]
        out = {k: self.f[k][vid] for k in self.mk}
        out["struct"] = self.f["struct"][vid]
        out["label"] = torch.tensor(self.lm[self.f["labels"][vid]["Label"]], dtype=torch.long)
        return out
def collate_fn(batch):
    return {k: torch.stack([b[k] for b in batch]) for k in batch[0]}

class FlexFusion(nn.Module):
    def __init__(self, mod_keys, mod_dim=768, hidden=192, num_heads=4, struct_dim=9, num_classes=2, dropout=0.15, mod_drop=0.15):
        super().__init__()
        self.mk = mod_keys; nm = len(mod_keys); self.h = hidden; self.md = mod_drop
        self.projs = nn.ModuleList([nn.Sequential(nn.Linear(mod_dim, hidden), nn.GELU(), nn.Dropout(dropout), nn.LayerNorm(hidden)) for _ in range(nm)])
        self.struct_enc = nn.Sequential(nn.Linear(struct_dim, 64), nn.GELU(), nn.LayerNorm(64))
        self.routes = nn.ModuleList([nn.Sequential(nn.Linear(hidden, hidden//2), nn.GELU(), nn.Linear(hidden//2, 1)) for _ in range(num_heads)])
        cd = num_heads * hidden + hidden + 64
        self.cls = nn.Sequential(nn.LayerNorm(cd), nn.Linear(cd, 256), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(256, 64), nn.GELU(), nn.Dropout(dropout*0.5), nn.Linear(64, num_classes))
    def forward(self, batch, training=False):
        refined = []
        for i, (proj, k) in enumerate(zip(self.projs, self.mk)):
            h = proj(batch[k])
            if training and self.md > 0: h = h * (torch.rand(h.size(0), 1, device=h.device) > self.md).float()
            refined.append(h)
        st = torch.stack(refined, dim=1)
        heads = [((st * torch.softmax(rm(st).squeeze(-1), dim=1).unsqueeze(-1)).sum(dim=1)) for rm in self.routes]
        return self.cls(torch.cat(heads + [st.mean(dim=1), self.struct_enc(batch["struct"])], dim=-1))

def cosine_warmup(opt, ws, ts):
    def f(s):
        if s < ws: return s/max(1,ws)
        return max(0, 0.5*(1+np.cos(np.pi*(s-ws)/max(1,ts-ws))))
    return LambdaLR(opt, f)

def load_split_ids(split_dir):
    s = {}
    for n in ["train","valid","test"]:
        with open(os.path.join(split_dir, f"{n}.csv")) as f: s[n]=[r[0] for r in csv.reader(f) if r]
    return s

def run_config(name, features, split_ids, label_map, mod_keys, struct_dim, hidden, num_heads, lr, epochs, dropout, mod_drop, num_runs, logger):
    feat_keys = list(mod_keys) + ["struct"]
    common = set.intersection(*[set(features[k].keys()) for k in feat_keys]) & set(features["labels"].keys())
    cur = {s: [v for v in split_ids[s] if v in common] for s in split_ids}
    accs = []
    for ri in range(num_runs):
        seed = ri*1000+42; torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
        trd = DS(cur["train"], features, label_map, mod_keys); vd = DS(cur["valid"], features, label_map, mod_keys); ted = DS(cur["test"], features, label_map, mod_keys)
        trl = DataLoader(trd, 32, True, collate_fn=collate_fn); vl = DataLoader(vd, 64, False, collate_fn=collate_fn); tel = DataLoader(ted, 64, False, collate_fn=collate_fn)
        model = FlexFusion(mod_keys, hidden=hidden, num_heads=num_heads, struct_dim=struct_dim, dropout=dropout, mod_drop=mod_drop).to(device)
        ema = copy.deepcopy(model)
        opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.02)
        ts = epochs*len(trl); ws = 5*len(trl); sch = cosine_warmup(opt, ws, ts)
        crit = nn.CrossEntropyLoss(label_smoothing=0.03); bva, bst = -1, None
        for ep in range(epochs):
            model.train()
            for batch in trl:
                batch = {k:v.to(device) for k,v in batch.items()}; opt.zero_grad()
                crit(model(batch, training=True), batch["label"]).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sch.step()
                with torch.no_grad():
                    for p, ep2 in zip(model.parameters(), ema.parameters()): ep2.data.mul_(0.999).add_(p.data, alpha=0.001)
            ema.eval(); ps, ls2 = [], []
            with torch.no_grad():
                for batch in vl:
                    batch = {k:v.to(device) for k,v in batch.items()}
                    ps.extend(ema(batch).argmax(1).cpu().numpy()); ls2.extend(batch["label"].cpu().numpy())
            va = accuracy_score(ls2, ps)
            if va > bva: bva = va; bst = {k:v.clone() for k,v in ema.state_dict().items()}
        ema.load_state_dict(bst); ema.eval(); ps, ls2 = [], []
        with torch.no_grad():
            for batch in tel:
                batch = {k:v.to(device) for k,v in batch.items()}
                ps.extend(ema(batch).argmax(1).cpu().numpy()); ls2.extend(batch["label"].cpu().numpy())
        accs.append(accuracy_score(ls2, ps))
    logger.info(f"  {name}: Acc={np.mean(accs):.4f}±{np.std(accs):.4f} (max={np.max(accs):.4f}) >=0.85:{sum(1 for a in accs if a>=0.85)} >=0.90:{sum(1 for a in accs if a>=0.90)}")
    return np.mean(accs), np.max(accs)

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
    with open(ann_path) as f: features["labels"] = {d["Video_ID"]:d for d in json.load(f)}
    split_ids = load_split_ids(split_dir)
    struct_dim = features["struct"][list(features["struct"].keys())[0]].shape[0]
    logger.info(f"{args.dataset_name} {args.language}")

    nr = args.num_runs
    all_mods = ["text", "audio", "frame", "t1", "t2"]
    text_mods = ["text", "t1", "t2"]

    # Baseline
    run_config("baseline h=192 nh=4 lr=2e-4 ep=45", features, split_ids, label_map, all_mods, struct_dim, 192, 4, 2e-4, 45, 0.15, 0.15, nr, logger)

    # Hidden size sweep
    for h in [128, 256, 384]:
        run_config(f"h={h}", features, split_ids, label_map, all_mods, struct_dim, h, 4, 2e-4, 45, 0.15, 0.15, nr, logger)

    # LR sweep
    for lr in [1e-4, 5e-4, 1e-3]:
        run_config(f"lr={lr}", features, split_ids, label_map, all_mods, struct_dim, 192, 4, lr, 45, 0.15, 0.15, nr, logger)

    # Heads sweep
    for nh in [2, 6, 8]:
        run_config(f"nh={nh}", features, split_ids, label_map, all_mods, struct_dim, 192, nh, 2e-4, 45, 0.15, 0.15, nr, logger)

    # Epochs sweep
    for ep in [60, 80, 100]:
        run_config(f"ep={ep}", features, split_ids, label_map, all_mods, struct_dim, 192, 4, 2e-4, ep, 0.15, 0.15, nr, logger)

    # Dropout sweep
    for d in [0.1, 0.2, 0.3]:
        run_config(f"drop={d}", features, split_ids, label_map, all_mods, struct_dim, 192, 4, 2e-4, 45, d, d, nr, logger)

    # Text-only
    run_config("text-only", features, split_ids, label_map, text_mods, struct_dim, 192, 4, 2e-4, 45, 0.15, 0.15, nr, logger)

    # Text-only with different params
    run_config("text-only h=256 ep=60", features, split_ids, label_map, text_mods, struct_dim, 256, 4, 2e-4, 60, 0.15, 0.15, nr, logger)

if __name__ == "__main__":
    main()
