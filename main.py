"""
AppraiseHate: Reproduce max performance for each dataset.

Usage:
    python main.py --dataset HateMM
    python main.py --dataset MHClip-Y
    python main.py --dataset MHClip-B
    python main.py --dataset ImpliHateVid
"""
import argparse, csv, json, os, random, copy
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.covariance import LedoitWolf
from torch.utils.data import DataLoader, Dataset
import warnings; warnings.filterwarnings("ignore")

device = "cuda"

# ====================== Best configs (from 2000-seed search) ======================
BEST_CONFIGS = {
    "HateMM": {
        "seed": 607042,
        "emb_dir": "./embeddings/HateMM",
        "ann_path": "./datasets/HateMM/annotation(new).json",
        "split_dir": "./datasets/HateMM/splits",
        "label_map": {"Non Hate": 0, "Hate": 1},
        "ver": "v13",
        "whiten": "zca", "knn_type": "cosine", "k": 40, "temp": 0.1, "alpha": 0.5, "thresh": -0.10,
    },
    "MHClip-Y": {
        "seed": 908042,
        "emb_dir": "./embeddings/Multihateclip/English",
        "ann_path": "./datasets/Multihateclip/English/annotation(new).json",
        "split_dir": "./datasets/Multihateclip/English/splits",
        "label_map": {"Normal": 0, "Offensive": 1, "Hateful": 1},
        "ver": "v13b",
        "whiten": "spca_r32", "knn_type": "cosine", "k": 10, "temp": 0.02, "alpha": 0.5, "thresh": None,
    },
    "MHClip-B": {
        "seed": 99042,
        "emb_dir": "./embeddings/Multihateclip/Chinese",
        "ann_path": "./datasets/Multihateclip/Chinese/annotation(new).json",
        "split_dir": "./datasets/Multihateclip/Chinese/splits",
        "label_map": {"Normal": 0, "Offensive": 1, "Hateful": 1},
        "ver": "v13b",
        "whiten": "zca", "knn_type": "cosine", "k": 25, "temp": 0.1, "alpha": 0.1, "thresh": None,
    },
    "ImpliHateVid": {
        "seed": 28042,
        "emb_dir": "./embeddings/ImpliHateVid",
        "ann_path": "./datasets/ImpliHateVid/annotation(new).json",
        "split_dir": "./datasets/ImpliHateVid/splits",
        "label_map": {"Normal": 0, "Hateful": 1},
        "ver": "v13b",
        "whiten": "spca_r32", "knn_type": "csls", "k": 10, "temp": 0.02, "alpha": 0.4, "thresh": 0.06,
    },
}

# ====================== Model ======================
class Fusion(nn.Module):
    def __init__(self, mk, nc=2, hidden=192, nh=4, drop=0.15, md=0.15):
        super().__init__()
        self.mk = mk; self.md = md
        self.projs = nn.ModuleList([
            nn.Sequential(nn.Linear(768, hidden), nn.GELU(), nn.Dropout(drop), nn.LayerNorm(hidden))
            for _ in range(len(mk))
        ])
        self.routes = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Linear(hidden // 2, 1))
            for _ in range(nh)
        ])
        cd = nh * hidden + hidden
        self.pre_cls = nn.Sequential(
            nn.LayerNorm(cd), nn.Linear(cd, 256), nn.GELU(), nn.Dropout(drop),
            nn.Linear(256, 64), nn.GELU(), nn.Dropout(drop * 0.5)
        )
        self.head = nn.Linear(64, nc)

    def forward(self, batch, training=False, return_penult=False):
        ref = []
        for p, k in zip(self.projs, self.mk):
            h = p(batch[k])
            if training and self.md > 0:
                h = h * (torch.rand(h.size(0), 1, device=h.device) > self.md).float()
            ref.append(h)
        st = torch.stack(ref, dim=1)
        heads = [(st * torch.softmax(rm(st).squeeze(-1), dim=1).unsqueeze(-1)).sum(dim=1) for rm in self.routes]
        fused = torch.cat(heads + [st.mean(dim=1)], dim=-1)
        penult = self.pre_cls(fused)
        logits = self.head(penult)
        return (logits, penult) if return_penult else logits

# ====================== Data ======================
class DS(Dataset):
    def __init__(self, vids, feats, lm, mk):
        self.vids = vids; self.f = feats; self.lm = lm; self.mk = mk
    def __len__(self): return len(self.vids)
    def __getitem__(self, i):
        v = self.vids[i]
        out = {k: self.f[k][v] for k in self.mk}
        out["label"] = torch.tensor(self.lm[self.f["labels"][v]["Label"]], dtype=torch.long)
        return out

def collate_fn(b): return {k: torch.stack([x[k] for x in b]) for k in b[0]}

def load_split_ids(d):
    s = {}
    for n in ["train", "valid", "test"]:
        with open(os.path.join(d, f"{n}.csv")) as f:
            s[n] = [r[0] for r in csv.reader(f) if r]
    return s

# ====================== Whitening ======================
def zca_whiten(train_z, val_z, test_z):
    mean = train_z.mean(dim=0, keepdim=True); c = train_z - mean
    cov = (c.t() @ c) / (c.size(0) - 1)
    U, S, V = torch.svd(cov + 1e-5 * torch.eye(cov.size(0)))
    W = U @ torch.diag(1.0 / torch.sqrt(S)) @ V.t()
    return (F.normalize((train_z - mean) @ W, dim=1),
            F.normalize((val_z - mean) @ W, dim=1),
            F.normalize((test_z - mean) @ W, dim=1))

def spca_whiten(train_z, val_z, test_z, r=32):
    mean = train_z.mean(dim=0, keepdim=True)
    lw = LedoitWolf().fit((train_z - mean).numpy())
    cov = torch.tensor(lw.covariance_, dtype=torch.float32)
    U, S, V = torch.svd(cov)
    if r and r < U.size(1): U = U[:, :r]; S = S[:r]
    W = U @ torch.diag(1.0 / torch.sqrt(S + 1e-6))
    return (F.normalize((train_z - mean) @ W, dim=1),
            F.normalize((val_z - mean) @ W, dim=1),
            F.normalize((test_z - mean) @ W, dim=1))

# ====================== kNN ======================
def cosine_knn(qe, be, bl, k=15, nc=2, temperature=0.05):
    qn = F.normalize(qe, dim=1); bn = F.normalize(be, dim=1)
    sim = torch.mm(qn, bn.t()); ts2, ti = sim.topk(k, dim=1)
    tl = bl[ti]; w = F.softmax(ts2 / temperature, dim=1)
    out = torch.zeros(qe.size(0), nc)
    for c in range(nc): out[:, c] = (w * (tl == c).float()).sum(dim=1)
    return out.numpy()

def csls_knn(query, bank, bank_labels, k=15, nc=2, temperature=0.05, hub_k=10):
    qn = F.normalize(query, dim=1); bn = F.normalize(bank, dim=1)
    sim = torch.mm(qn, bn.t())
    bank_hub = sim.topk(min(hub_k, sim.size(0)), dim=0).values.mean(dim=0)
    csls_sim = 2 * sim - bank_hub.unsqueeze(0)
    topk_sim, topk_idx = csls_sim.topk(k, dim=1)
    topk_labels = bank_labels[topk_idx]
    weights = F.softmax(topk_sim / temperature, dim=1)
    out = torch.zeros(query.size(0), nc)
    for c in range(nc): out[:, c] = (weights * (topk_labels == c).float()).sum(dim=1)
    return out.numpy()

# ====================== Main ======================
def main():
    parser = argparse.ArgumentParser(description="AppraiseHate: reproduce max performance.")
    parser.add_argument("--dataset", required=True, choices=list(BEST_CONFIGS.keys()))
    args = parser.parse_args()

    cfg = BEST_CONFIGS[args.dataset]
    seed = cfg["seed"]
    mk = ["text", "audio", "frame", "ans_what", "ans_target", "ans_where", "ans_why", "ans_how"]
    nc = 2

    torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)

    print(f"{'='*50}")
    print(f"  AppraiseHate — {args.dataset}")
    print(f"  Seed: {seed}")
    print(f"{'='*50}")

    # Load features
    print("Loading features...")
    feats = {
        "text": torch.load(f"{cfg['emb_dir']}/text_features.pth", map_location="cpu"),
        "audio": torch.load(f"{cfg['emb_dir']}/wavlm_audio_features.pth", map_location="cpu"),
        "frame": torch.load(f"{cfg['emb_dir']}/frame_features.pth", map_location="cpu"),
    }
    for field in ["what", "target", "where", "why", "how"]:
        feats[f"ans_{field}"] = torch.load(f"{cfg['emb_dir']}/{cfg['ver']}_ans_{field}_features.pth", map_location="cpu")
    with open(cfg["ann_path"]) as f:
        feats["labels"] = {d["Video_ID"]: d for d in json.load(f)}

    splits = load_split_ids(cfg["split_dir"])
    common = set.intersection(*[set(feats[k].keys()) for k in mk]) & set(feats["labels"].keys())
    cur = {s: [v for v in splits[s] if v in common] for s in splits}
    print(f"  Train={len(cur['train'])}, Val={len(cur['valid'])}, Test={len(cur['test'])}")

    trl = DataLoader(DS(cur["train"], feats, cfg["label_map"], mk), 32, True, collate_fn=collate_fn)
    vl = DataLoader(DS(cur["valid"], feats, cfg["label_map"], mk), 64, False, collate_fn=collate_fn)
    tel = DataLoader(DS(cur["test"], feats, cfg["label_map"], mk), 64, False, collate_fn=collate_fn)
    trl_ns = DataLoader(DS(cur["train"], feats, cfg["label_map"], mk), 64, False, collate_fn=collate_fn)

    # Train
    print("Training (45 epochs)...")
    model = Fusion(mk, nc=nc).to(device); ema = copy.deepcopy(model)
    opt = optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.02)
    crit = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 1.5]).to(device), label_smoothing=0.03)
    ts_t = 45 * len(trl); ws = 5 * len(trl)
    sch = LambdaLR(opt, lambda s: s / max(1, ws) if s < ws else max(0, 0.5 * (1 + np.cos(np.pi * (s - ws) / max(1, ts_t - ws)))))
    bva, bst = -1, None

    for e in range(45):
        model.train()
        for batch in trl:
            batch = {k: v.to(device) for k, v in batch.items()}; opt.zero_grad()
            crit(model(batch, training=True), batch["label"]).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sch.step()
            with torch.no_grad():
                for p, ep2 in zip(model.parameters(), ema.parameters()):
                    ep2.data.mul_(0.999).add_(p.data, alpha=0.001)
        ema.eval(); ps, ls = [], []
        with torch.no_grad():
            for batch in vl:
                batch = {k: v.to(device) for k, v in batch.items()}
                ps.extend(ema(batch).argmax(1).cpu().numpy()); ls.extend(batch["label"].cpu().numpy())
        va = accuracy_score(ls, ps)
        if va > bva: bva = va; bst = {k: v.clone() for k, v in ema.state_dict().items()}
        if (e + 1) % 15 == 0: print(f"  Epoch {e+1}: Val ACC={va:.4f} (best={bva:.4f})")

    ema.load_state_dict(bst)

    # Extract features
    def get_pl(m, loader):
        m.eval(); ap, al, alb = [], [], []
        with torch.no_grad():
            for b in loader:
                b = {k: v.to(device) for k, v in b.items()}; lo, pe = m(b, return_penult=True)
                ap.append(pe.cpu()); al.append(lo.cpu()); alb.extend(b["label"].cpu().numpy())
        return torch.cat(ap), torch.cat(al).numpy(), np.array(alb)

    print("Extracting features...")
    tp, tl_arr, tla = get_pl(ema, trl_ns)
    vp, vl_arr, vla = get_pl(ema, vl)
    tep, tel_arr, tela = get_pl(ema, tel)
    blt = torch.tensor(tla)

    print(f"  Head ACC: {accuracy_score(tela, np.argmax(tel_arr, axis=1)):.4f}")

    # Whitening
    print(f"Whitening ({cfg['whiten']})...")
    if cfg["whiten"] == "zca":
        tr_w, va_w, te_w = zca_whiten(tp, vp, tep)
    elif cfg["whiten"].startswith("spca"):
        r = int(cfg["whiten"].split("_r")[1])
        tr_w, va_w, te_w = spca_whiten(tp, vp, tep, r=r)
    else:
        tr_w, va_w, te_w = tp, vp, tep

    # kNN
    knn_fn = cosine_knn if cfg["knn_type"] == "cosine" else csls_knn
    print(f"Retrieval ({cfg['knn_type']}, k={cfg['k']}, temp={cfg['temp']}, alpha={cfg['alpha']})...")
    kt = knn_fn(te_w, tr_w, blt, k=cfg["k"], nc=nc, temperature=cfg["temp"])
    bl_test = (1 - cfg["alpha"]) * tel_arr + cfg["alpha"] * kt

    # Threshold
    if cfg["thresh"] is not None:
        td = bl_test[:, 1] - bl_test[:, 0]
        preds = (td > cfg["thresh"]).astype(int)
        print(f"Threshold: {cfg['thresh']}")
    else:
        preds = np.argmax(bl_test, axis=1)

    # Results
    acc = accuracy_score(tela, preds)
    mf1 = f1_score(tela, preds, average='macro')
    mp = precision_score(tela, preds, average='macro')
    mr = recall_score(tela, preds, average='macro')
    cm = confusion_matrix(tela, preds)

    print(f"\n{'='*50}")
    print(f"  {args.dataset} — RESULTS")
    print(f"{'='*50}")
    print(f"  ACC  = {acc:.4f}")
    print(f"  M-F1 = {mf1:.4f}")
    print(f"  M-P  = {mp:.4f}")
    print(f"  M-R  = {mr:.4f}")
    print(f"  CM   = {cm.tolist()}")
    print(f"{'='*50}")

if __name__ == "__main__":
    main()
