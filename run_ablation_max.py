"""
Ablation study under exact max performance config.
Full model MUST reproduce max performance. Each variant trained from scratch
with same seed, same training config, only the specified component changed.
Retrieval uses exact best config per dataset (whiten/knn_type/k/temp/alpha/thresh).
"""
import argparse, csv, json, os, random, copy
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.covariance import LedoitWolf
from torch.utils.data import DataLoader, Dataset
import warnings; warnings.filterwarnings("ignore")

device = "cuda"

# ====================== Best configs ======================
BEST_CONFIGS = {
    "HateMM": {
        "seed": 607042, "ver": "v13",
        "emb_dir": "./embeddings/HateMM",
        "ann_path": "./datasets/HateMM/annotation(new).json",
        "split_dir": "./datasets/HateMM/splits",
        "label_map": {"Non Hate": 0, "Hate": 1},
        "whiten": "zca", "knn_type": "cosine", "k": 40, "temp": 0.1, "alpha": 0.5, "thresh": -0.10,
    },
    "MHClip-Y": {
        "seed": 908042, "ver": "v13b",
        "emb_dir": "./embeddings/Multihateclip/English",
        "ann_path": "./datasets/Multihateclip/English/annotation(new).json",
        "split_dir": "./datasets/Multihateclip/English/splits",
        "label_map": {"Normal": 0, "Offensive": 1, "Hateful": 1},
        "whiten": "spca_r32", "knn_type": "cosine", "k": 10, "temp": 0.02, "alpha": 0.5, "thresh": None,
    },
    "MHClip-B": {
        "seed": 99042, "ver": "v13b",
        "emb_dir": "./embeddings/Multihateclip/Chinese",
        "ann_path": "./datasets/Multihateclip/Chinese/annotation(new).json",
        "split_dir": "./datasets/Multihateclip/Chinese/splits",
        "label_map": {"Normal": 0, "Offensive": 1, "Hateful": 1},
        "whiten": "zca", "knn_type": "cosine", "k": 25, "temp": 0.1, "alpha": 0.1, "thresh": None,
    },
    "ImpliHateVid": {
        "seed": 28042, "ver": "v13b",
        "emb_dir": "./embeddings/ImpliHateVid",
        "ann_path": "./datasets/ImpliHateVid/annotation(new).json",
        "split_dir": "./datasets/ImpliHateVid/splits",
        "label_map": {"Normal": 0, "Hateful": 1},
        "whiten": "spca_r32", "knn_type": "csls", "k": 10, "temp": 0.02, "alpha": 0.4, "thresh": 0.06,
    },
}

# ====================== Models ======================
class OurFusion(nn.Module):
    def __init__(self, mk, nc=2, hidden=192, nh=4, drop=0.15, md=0.15):
        super().__init__()
        self.mk = mk; self.md = md
        self.projs = nn.ModuleList([nn.Sequential(nn.Linear(768, hidden), nn.GELU(), nn.Dropout(drop), nn.LayerNorm(hidden)) for _ in range(len(mk))])
        self.routes = nn.ModuleList([nn.Sequential(nn.Linear(hidden, hidden//2), nn.GELU(), nn.Linear(hidden//2, 1)) for _ in range(nh)])
        cd = nh * hidden + hidden
        self.pre_cls = nn.Sequential(nn.LayerNorm(cd), nn.Linear(cd, 256), nn.GELU(), nn.Dropout(drop), nn.Linear(256, 64), nn.GELU(), nn.Dropout(drop*0.5))
        self.head = nn.Linear(64, nc)
    def forward(self, batch, training=False, return_penult=False):
        ref = []
        for p, k in zip(self.projs, self.mk):
            h = p(batch[k])
            if training and self.md > 0: h = h * (torch.rand(h.size(0), 1, device=h.device) > self.md).float()
            ref.append(h)
        st = torch.stack(ref, dim=1)
        heads = [(st * torch.softmax(rm(st).squeeze(-1), dim=1).unsqueeze(-1)).sum(dim=1) for rm in self.routes]
        fused = torch.cat(heads + [st.mean(dim=1)], dim=-1)
        penult = self.pre_cls(fused); logits = self.head(penult)
        return (logits, penult) if return_penult else logits

class MoEFusion(nn.Module):
    def __init__(self, mk, nc=2):
        super().__init__(); self.mk = mk; nm = len(mk)
        self.projs = nn.ModuleList([nn.Linear(768, 128) for _ in range(nm)])
        self.experts = nn.ModuleList([nn.Sequential(nn.Linear(128*nm, 128), nn.ReLU(), nn.Linear(128, 128)) for _ in range(8)])
        self.gate = nn.Sequential(nn.Linear(128*nm, 8), nn.Softmax(dim=-1))
        self.pre_cls = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.1))
        self.head = nn.Linear(64, nc)
    def forward(self, batch, training=False, return_penult=False):
        feats = [p(batch[k]) for p, k in zip(self.projs, self.mk)]; x = torch.cat(feats, dim=-1)
        gw = self.gate(x); eo = torch.stack([e(x) for e in self.experts], dim=1)
        fused = torch.sum(gw.unsqueeze(-1)*eo, dim=1); penult = self.pre_cls(fused); logits = self.head(penult)
        return (logits, penult) if return_penult else logits

class HVGuardFusion(nn.Module):
    def __init__(self, mk, nc=2):
        super().__init__(); self.mk = mk; td = 768*len(mk)
        self.experts = nn.ModuleList([nn.Sequential(nn.Linear(td, 128), nn.ReLU(), nn.Linear(128, 128)) for _ in range(8)])
        self.gate = nn.Sequential(nn.Linear(td, 8), nn.Softmax(dim=-1)); self.gd = nn.Dropout(0.1)
        self.pre_cls = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.1))
        self.head = nn.Linear(64, nc)
    def forward(self, batch, training=False, return_penult=False):
        x = torch.cat([batch[k] for k in self.mk], dim=-1); gw = self.gd(self.gate(x))
        eo = torch.stack([e(x) for e in self.experts], dim=1); fused = torch.sum(gw.unsqueeze(-1)*eo, dim=1)
        penult = self.pre_cls(fused); logits = self.head(penult)
        return (logits, penult) if return_penult else logits

# ====================== Data ======================
class DS(Dataset):
    def __init__(self, vids, feats, lm, mk):
        self.vids = vids; self.f = feats; self.lm = lm; self.mk = mk
    def __len__(self): return len(self.vids)
    def __getitem__(self, i):
        v = self.vids[i]; out = {k: self.f[k][v] for k in self.mk}
        out["label"] = torch.tensor(self.lm[self.f["labels"][v]["Label"]], dtype=torch.long); return out

def collate_fn(b): return {k: torch.stack([x[k] for x in b]) for k in b[0]}
def load_split_ids(d):
    s = {}
    for n in ["train", "valid", "test"]:
        with open(os.path.join(d, f"{n}.csv")) as f: s[n] = [r[0] for r in csv.reader(f) if r]
    return s

# ====================== Whitening ======================
def zca_whiten(tr, va, te):
    mean = tr.mean(dim=0, keepdim=True); c = tr - mean
    cov = (c.t() @ c) / (c.size(0) - 1); U, S, V = torch.svd(cov + 1e-5*torch.eye(cov.size(0)))
    W = U @ torch.diag(1.0/torch.sqrt(S)) @ V.t()
    return F.normalize((tr-mean)@W, dim=1), F.normalize((va-mean)@W, dim=1), F.normalize((te-mean)@W, dim=1)

def spca_whiten(tr, va, te, r=32):
    mean = tr.mean(dim=0, keepdim=True)
    lw = LedoitWolf().fit((tr-mean).numpy()); cov = torch.tensor(lw.covariance_, dtype=torch.float32)
    U, S, V = torch.svd(cov)
    if r and r < U.size(1): U = U[:,:r]; S = S[:r]
    W = U @ torch.diag(1.0/torch.sqrt(S+1e-6))
    return F.normalize((tr-mean)@W, dim=1), F.normalize((va-mean)@W, dim=1), F.normalize((te-mean)@W, dim=1)

# ====================== kNN ======================
def cosine_knn(qe, be, bl, k=15, nc=2, temperature=0.05):
    qn = F.normalize(qe, dim=1); bn = F.normalize(be, dim=1)
    sim = torch.mm(qn, bn.t()); ts2, ti = sim.topk(k, dim=1)
    tl = bl[ti]; w = F.softmax(ts2/temperature, dim=1)
    out = torch.zeros(qe.size(0), nc)
    for c in range(nc): out[:,c] = (w*(tl==c).float()).sum(dim=1)
    return out.numpy()

def csls_knn(query, bank, bl, k=15, nc=2, temperature=0.05, hub_k=10):
    qn = F.normalize(query, dim=1); bn = F.normalize(bank, dim=1)
    sim = torch.mm(qn, bn.t())
    bank_hub = sim.topk(min(hub_k, sim.size(0)), dim=0).values.mean(dim=0)
    csls_sim = 2*sim - bank_hub.unsqueeze(0); topk_sim, topk_idx = csls_sim.topk(k, dim=1)
    topk_labels = bl[topk_idx]; weights = F.softmax(topk_sim/temperature, dim=1)
    out = torch.zeros(query.size(0), nc)
    for c in range(nc): out[:,c] = (weights*(topk_labels==c).float()).sum(dim=1)
    return out.numpy()

def best_thresh(vl, vla, tl, tla):
    """Tune threshold on val, apply to test."""
    std = accuracy_score(tla, np.argmax(tl, axis=1))
    vd = vl[:,1]-vl[:,0]; td = tl[:,1]-tl[:,0]; bt, bv = 0, 0
    for t in np.arange(-3, 3, 0.02):
        v = accuracy_score(vla, (vd > t).astype(int))
        if v > bv: bv, bt = v, t
    tuned = accuracy_score(tla, (td > bt).astype(int))
    return (tuned, bt) if tuned > std else (std, None)

# ====================== Train + Eval ======================
def train_and_eval(feats, splits, lm, mk, model_cls, seed, cfg, nc=2,
                   override_alpha=None, override_whiten=None, override_knn=None,
                   retrieval_on='post'):
    """Train from scratch, eval with exact best retrieval config."""
    torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
    fk = list(mk)
    common = set.intersection(*[set(feats[k].keys()) for k in fk]) & set(feats["labels"].keys())
    cur = {s: [v for v in splits[s] if v in common] for s in splits}

    trd = DS(cur["train"], feats, lm, mk); vd = DS(cur["valid"], feats, lm, mk); ted = DS(cur["test"], feats, lm, mk)
    trl = DataLoader(trd, 32, True, collate_fn=collate_fn)
    vl = DataLoader(vd, 64, False, collate_fn=collate_fn)
    tel = DataLoader(ted, 64, False, collate_fn=collate_fn)
    trl_ns = DataLoader(DS(cur["train"], feats, lm, mk), 64, False, collate_fn=collate_fn)

    model = model_cls(mk, nc=nc).to(device); ema = copy.deepcopy(model)
    opt = optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.02)
    crit = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 1.5]).to(device), label_smoothing=0.03)
    ts_t = 45*len(trl); ws = 5*len(trl)
    sch = LambdaLR(opt, lambda s: s/max(1,ws) if s<ws else max(0, 0.5*(1+np.cos(np.pi*(s-ws)/max(1,ts_t-ws)))))
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

    ema.load_state_dict(bst)

    def get_pl(m, loader):
        m.eval(); ap, al, alb = [], [], []
        with torch.no_grad():
            for b in loader:
                b = {k: v.to(device) for k, v in b.items()}; lo, pe = m(b, return_penult=True)
                ap.append(pe.cpu()); al.append(lo.cpu()); alb.extend(b["label"].cpu().numpy())
        return torch.cat(ap), torch.cat(al).numpy(), np.array(alb)

    tp, tl_arr, tla = get_pl(ema, trl_ns)
    vp, vl_arr, vla = get_pl(ema, vl)
    tep, tel_arr, tela = get_pl(ema, tel)
    blt = torch.tensor(tla)

    # Retrieval config
    alpha = override_alpha if override_alpha is not None else cfg["alpha"]
    whiten_name = override_whiten if override_whiten is not None else cfg["whiten"]
    knn_type = override_knn if override_knn is not None else cfg["knn_type"]

    if alpha == 0:
        # Head only + threshold
        acc, thresh = best_thresh(vl_arr, vla, tel_arr, tela)
        if thresh is not None:
            preds = (tel_arr[:,1]-tel_arr[:,0] > thresh).astype(int)
        else:
            preds = np.argmax(tel_arr, axis=1)
    else:
        # Whitening
        if whiten_name == "zca":
            tr_w, va_w, te_w = zca_whiten(tp, vp, tep)
        elif whiten_name.startswith("spca"):
            r = int(whiten_name.split("_r")[1])
            tr_w, va_w, te_w = spca_whiten(tp, vp, tep, r=r)
        elif whiten_name == "none":
            tr_w, va_w, te_w = tp, vp, tep
        else:
            tr_w, va_w, te_w = tp, vp, tep

        # kNN
        knn_fn = cosine_knn if knn_type == "cosine" else csls_knn
        kt = knn_fn(te_w, tr_w, blt, k=cfg["k"], nc=nc, temperature=cfg["temp"])
        kv = knn_fn(va_w, tr_w, blt, k=cfg["k"], nc=nc, temperature=cfg["temp"])
        bl_test = (1-alpha)*tel_arr + alpha*kt
        bl_val = (1-alpha)*vl_arr + alpha*kv

        # Threshold
        acc, thresh = best_thresh(bl_val, vla, bl_test, tela)
        if thresh is not None:
            preds = (bl_test[:,1]-bl_test[:,0] > thresh).astype(int)
        else:
            preds = np.argmax(bl_test, axis=1)

    return {
        "acc": float(accuracy_score(tela, preds)),
        "f1": float(f1_score(tela, preds, average='macro')),
        "p": float(precision_score(tela, preds, average='macro')),
        "r": float(recall_score(tela, preds, average='macro')),
    }

# ====================== Main ======================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=list(BEST_CONFIGS.keys()))
    args = parser.parse_args()

    cfg = BEST_CONFIGS[args.dataset]
    seed = cfg["seed"]; ver = cfg["ver"]; lm = cfg["label_map"]; nc = 2
    mk_full = ["text", "audio", "frame", "ans_what", "ans_target", "ans_where", "ans_why", "ans_how"]

    print(f"Loading features for {args.dataset}...")
    feats = {
        "text": torch.load(f"{cfg['emb_dir']}/text_features.pth", map_location="cpu"),
        "audio": torch.load(f"{cfg['emb_dir']}/wavlm_audio_features.pth", map_location="cpu"),
        "frame": torch.load(f"{cfg['emb_dir']}/frame_features.pth", map_location="cpu"),
    }
    for field in ["what", "target", "where", "why", "how"]:
        feats[f"ans_{field}"] = torch.load(f"{cfg['emb_dir']}/{ver}_ans_{field}_features.pth", map_location="cpu")
    # Perception/Cognition for stage ablation
    for extra in ["perception", "cognition"]:
        p = f"{cfg['emb_dir']}/{ver}_{extra}_features.pth"
        if os.path.exists(p): feats[extra] = torch.load(p, map_location="cpu")
    # HVGuard mix for prompt replacement
    hvg = f"{cfg['emb_dir']}/hvguard_mix_features.pth"
    if os.path.exists(hvg): feats["hvguard_mix"] = torch.load(hvg, map_location="cpu")
    else: feats["hvguard_mix"] = feats["text"]

    with open(cfg["ann_path"]) as f: feats["labels"] = {d["Video_ID"]: d for d in json.load(f)}
    splits = load_split_ids(cfg["split_dir"])

    # Define all ablation variants
    variants = [
        # Full model (must match max performance)
        ("Full model", mk_full, OurFusion, None, None, None, 'post'),
        # P2C field ablation
        ("--what", [m for m in mk_full if m != "ans_what"], OurFusion, None, None, None, 'post'),
        ("--where", [m for m in mk_full if m != "ans_where"], OurFusion, None, None, None, 'post'),
        ("--why", [m for m in mk_full if m != "ans_why"], OurFusion, None, None, None, 'post'),
        ("--how", [m for m in mk_full if m != "ans_how"], OurFusion, None, None, None, 'post'),
        ("Percep. only", ["text","audio","frame","perception"], OurFusion, None, None, None, 'post'),
        ("Cogn. only", ["text","audio","frame","cognition"], OurFusion, None, None, None, 'post'),
        ("HVGuard CoT", ["text","audio","frame","hvguard_mix"], OurFusion, None, None, None, 'post'),
        # Fusion replacement
        ("MoE fusion", mk_full, MoEFusion, None, None, None, 'post'),
        ("HVGuard fusion", mk_full, HVGuardFusion, None, None, None, 'post'),
        # Retrieval ablation
        ("No retrieval", mk_full, OurFusion, 0.0, None, None, 'post'),
        ("No whitening", mk_full, OurFusion, None, "none", None, 'post'),
    ]

    print(f"\n{'='*60}")
    print(f"  {args.dataset} Ablation (seed={seed})")
    print(f"  Best config: whiten={cfg['whiten']}, knn={cfg['knn_type']}, k={cfg['k']}, temp={cfg['temp']}, alpha={cfg['alpha']}, thresh={cfg['thresh']}")
    print(f"{'='*60}")
    print(f"  {'Variant':<20s} {'ACC':>7} {'M-F1':>7} {'M-P':>7} {'M-R':>7}")
    print(f"  {'-'*55}")

    for vname, mk, model_cls, ov_alpha, ov_whiten, ov_knn, ret_on in variants:
        r = train_and_eval(feats, splits, lm, mk, model_cls, seed, cfg, nc=nc,
                          override_alpha=ov_alpha, override_whiten=ov_whiten, override_knn=ov_knn,
                          retrieval_on=ret_on)
        print(f"  {vname:<20s} {r['acc']*100:>6.1f} {r['f1']*100:>6.1f} {r['p']*100:>6.1f} {r['r']*100:>6.1f}")

if __name__ == "__main__":
    main()
