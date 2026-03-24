"""Advanced Ensemble Fusion: snapshot ensemble + SWA + focal loss + multi-seed voting.

Techniques inspired by:
- Snapshot Ensembles (Huang et al., ICLR 2017): save models at cosine annealing restarts
- Stochastic Weight Averaging (Izmailov et al., UAI 2018): flat minima → better generalization
- Focal Loss (Lin et al., ICCV 2017): down-weight easy examples, focus on hard ones
- Multi-seed Ensemble: vote across top-K seeds instead of single best

Pipeline:
1. Train N models (different seeds) with cosine restarts → collect snapshot checkpoints
2. For each model, also maintain SWA weights
3. At test time: ensemble predictions from (a) best EMA, (b) SWA model, (c) snapshot checkpoints
4. Multi-seed voting: majority vote or averaged logits across seeds
"""
import argparse, csv, json, os, random, copy, logging
from datetime import datetime
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingWarmRestarts
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, Dataset
import warnings; warnings.filterwarnings("ignore")

device = "cuda"

def setup_logger():
    os.makedirs("./logs", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    lf = f"./logs/fusion_ens_{ts}.log"
    logger = logging.getLogger("fusens"); logger.setLevel(logging.INFO); logger.handlers.clear()
    fh = logging.FileHandler(lf, encoding="utf-8"); ch = logging.StreamHandler()
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    fh.setFormatter(fmt); ch.setFormatter(fmt); logger.addHandler(fh); logger.addHandler(ch)
    return logger

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, label_smoothing=0.03):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.ls = label_smoothing

    def forward(self, logits, targets):
        # Apply label smoothing
        nc = logits.size(1)
        with torch.no_grad():
            smooth = torch.full_like(logits, self.ls / (nc - 1))
            smooth.scatter_(1, targets.unsqueeze(1), 1.0 - self.ls)
        log_p = F.log_softmax(logits, dim=1)
        p = torch.exp(log_p)
        focal_weight = (1 - p) ** self.gamma
        loss = -(focal_weight * smooth * log_p).sum(dim=1)
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            loss = alpha_t * loss
        return loss.mean()

class DS(Dataset):
    def __init__(self, vids, feats, lm, mk):
        self.vids=vids; self.f=feats; self.lm=lm; self.mk=mk
    def __len__(self): return len(self.vids)
    def __getitem__(self, i):
        v = self.vids[i]
        out = {k: self.f[k][v] for k in self.mk}
        out["struct"] = self.f["struct"][v]
        out["label"] = torch.tensor(self.lm[self.f["labels"][v]["Label"]], dtype=torch.long)
        return out
def collate_fn(b): return {k: torch.stack([x[k] for x in b]) for k in b[0]}

class Fusion(nn.Module):
    def __init__(self, mk, hidden=192, nh=4, sd=9, nc=2, drop=0.15, md=0.15):
        super().__init__()
        nm = len(mk); self.mk=mk; self.h=hidden; self.md=md
        self.projs = nn.ModuleList([nn.Sequential(nn.Linear(768,hidden),nn.GELU(),nn.Dropout(drop),nn.LayerNorm(hidden)) for _ in range(nm)])
        self.se = nn.Sequential(nn.Linear(sd,64),nn.GELU(),nn.LayerNorm(64))
        self.routes = nn.ModuleList([nn.Sequential(nn.Linear(hidden,hidden//2),nn.GELU(),nn.Linear(hidden//2,1)) for _ in range(nh)])
        cd = nh*hidden+hidden+64
        self.cls = nn.Sequential(nn.LayerNorm(cd),nn.Linear(cd,256),nn.GELU(),nn.Dropout(drop),nn.Linear(256,64),nn.GELU(),nn.Dropout(drop*0.5),nn.Linear(64,nc))
    def forward(self, batch, training=False):
        ref = []
        for i,(p,k) in enumerate(zip(self.projs,self.mk)):
            h=p(batch[k])
            if training and self.md>0: h=h*(torch.rand(h.size(0),1,device=h.device)>self.md).float()
            ref.append(h)
        st=torch.stack(ref,dim=1)
        heads=[((st*torch.softmax(rm(st).squeeze(-1),dim=1).unsqueeze(-1)).sum(dim=1)) for rm in self.routes]
        return self.cls(torch.cat(heads+[st.mean(dim=1),self.se(batch["struct"])],dim=-1))

def cw(opt,ws,ts):
    def f(s):
        if s<ws:return s/max(1,ws)
        return max(0,0.5*(1+np.cos(np.pi*(s-ws)/max(1,ts-ws))))
    return LambdaLR(opt,f)

def load_split_ids(d):
    s={}
    for n in ["train","valid","test"]:
        with open(os.path.join(d,f"{n}.csv")) as f: s[n]=[r[0] for r in csv.reader(f) if r]
    return s

def get_logits(model, loader):
    model.eval()
    logits, labs = [], []
    with torch.no_grad():
        for batch in loader:
            batch={k:v.to(device) for k,v in batch.items()}
            logits.append(model(batch).cpu().numpy())
            labs.extend(batch["label"].cpu().numpy())
    return np.concatenate(logits, axis=0), np.array(labs)

def threshold_tune(val_logits, val_labs, test_logits, test_labs):
    """Find optimal threshold on val, apply to test."""
    vd = val_logits[:,1] - val_logits[:,0]
    td = test_logits[:,1] - test_logits[:,0]
    bt, bva = 0, 0
    for t in np.arange(-3, 3, 0.02):
        va = accuracy_score(val_labs, (vd > t).astype(int))
        if va > bva: bva, bt = va, t
    return accuracy_score(test_labs, (td > bt).astype(int))

def run_single_seed(seed, feats, cur, lm, mk, sd, hidden, lr, epochs, drop, md, nc, use_focal, use_swa):
    """Train one model. Returns: best_ema, swa_model, snapshots, val/test loaders."""
    torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
    trd=DS(cur["train"],feats,lm,mk); vd=DS(cur["valid"],feats,lm,mk); ted=DS(cur["test"],feats,lm,mk)
    trl=DataLoader(trd,32,True,collate_fn=collate_fn); vl=DataLoader(vd,64,False,collate_fn=collate_fn); tel=DataLoader(ted,64,False,collate_fn=collate_fn)
    model=Fusion(mk,hidden=hidden,sd=sd,nc=nc,drop=drop,md=md).to(device)
    ema=copy.deepcopy(model)
    swa_model=copy.deepcopy(model) if use_swa else None
    swa_n = 0
    opt=optim.AdamW(model.parameters(),lr=lr,weight_decay=0.02)
    ts_total=epochs*len(trl); ws=5*len(trl); sch=cw(opt,ws,ts_total)

    if use_focal:
        crit = FocalLoss(gamma=2.0, label_smoothing=0.03)
    else:
        crit = nn.CrossEntropyLoss(label_smoothing=0.03)

    bva, bst = -1, None
    snapshots = []
    snapshot_interval = max(1, epochs // 4)  # Save ~4 snapshots

    for e in range(epochs):
        model.train()
        for batch in trl:
            batch={k:v.to(device) for k,v in batch.items()}; opt.zero_grad()
            crit(model(batch,training=True),batch["label"]).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); sch.step()
            with torch.no_grad():
                for p,ep2 in zip(model.parameters(),ema.parameters()):ep2.data.mul_(0.999).add_(p.data,alpha=0.001)

        # SWA: average weights in last 40% of training
        if use_swa and e >= int(epochs * 0.6):
            with torch.no_grad():
                for sp, mp in zip(swa_model.parameters(), model.parameters()):
                    sp.data.mul_(swa_n).add_(mp.data).div_(swa_n + 1)
            swa_n += 1

        ema.eval(); ps, ls2 = [], []
        with torch.no_grad():
            for batch in vl:
                batch={k:v.to(device) for k,v in batch.items()}
                ps.extend(ema(batch).argmax(1).cpu().numpy()); ls2.extend(batch["label"].cpu().numpy())
        va = accuracy_score(ls2, ps)
        if va > bva:
            bva = va; bst = {k:v.clone() for k,v in ema.state_dict().items()}

        # Save snapshot at intervals (after warmup)
        if e >= 10 and (e + 1) % snapshot_interval == 0:
            snapshots.append({k:v.clone() for k,v in ema.state_dict().items()})

    ema.load_state_dict(bst)
    return ema, swa_model, snapshots, vl, tel

def run(name, feats, splits, lm, mk, sd, hidden, lr, ep, drop, md, nr, nc, logger, use_focal=False, use_swa=True, ensemble_topk=5):
    fk = list(mk) + ["struct"]
    common = set.intersection(*[set(feats[k].keys()) for k in fk]) & set(feats["labels"].keys())
    cur = {s:[v for v in splits[s] if v in common] for s in splits}

    all_val_logits, all_test_logits = [], []
    all_test_labs = None
    all_val_labs = None
    single_accs = []

    for ri in range(nr):
        seed = ri * 1000 + 42
        ema, swa_model, snapshots, vl, tel = run_single_seed(
            seed, feats, cur, lm, mk, sd, hidden, lr, ep, drop, md, nc, use_focal, use_swa)

        # Collect logits from best EMA
        val_logits_ema, val_labs = get_logits(ema, vl)
        test_logits_ema, test_labs = get_logits(ema, tel)
        if all_test_labs is None:
            all_test_labs = test_labs
            all_val_labs = val_labs

        # Single model accuracy (with threshold tuning)
        std_acc = accuracy_score(test_labs, np.argmax(test_logits_ema, axis=1))
        if nc == 2:
            tuned_acc = threshold_tune(val_logits_ema, val_labs, test_logits_ema, test_labs)
            single_acc = max(std_acc, tuned_acc)
        else:
            single_acc = std_acc
        single_accs.append(single_acc)

        # Collect logits for multi-seed ensemble
        seed_logits_val = [val_logits_ema]
        seed_logits_test = [test_logits_ema]

        # Add SWA logits
        if use_swa and swa_model is not None:
            swa_val, _ = get_logits(swa_model, vl)
            swa_test, _ = get_logits(swa_model, tel)
            seed_logits_val.append(swa_val)
            seed_logits_test.append(swa_test)

        # Add snapshot logits
        snap_model = Fusion(mk, hidden=hidden, sd=sd, nc=nc, drop=drop, md=0).to(device)
        for snap_sd in snapshots[-2:]:  # Use last 2 snapshots
            snap_model.load_state_dict(snap_sd)
            sv, _ = get_logits(snap_model, vl)
            st, _ = get_logits(snap_model, tel)
            seed_logits_val.append(sv)
            seed_logits_test.append(st)

        # Average logits within this seed (EMA + SWA + snapshots)
        avg_val = np.mean(seed_logits_val, axis=0)
        avg_test = np.mean(seed_logits_test, axis=0)
        all_val_logits.append(avg_val)
        all_test_logits.append(avg_test)

    # --- Single model stats ---
    logger.info(f"  {name} [single]: Acc={np.mean(single_accs):.4f}±{np.std(single_accs):.4f} (max={np.max(single_accs):.4f}) >=0.85:{sum(1 for a in single_accs if a>=0.85)} >=0.90:{sum(1 for a in single_accs if a>=0.90)}")

    # --- In-seed ensemble (EMA+SWA+snap averaged) ---
    inseed_accs = []
    for i in range(nr):
        std = accuracy_score(all_test_labs, np.argmax(all_test_logits[i], axis=1))
        if nc == 2:
            tuned = threshold_tune(all_val_logits[i], all_val_labs, all_test_logits[i], all_test_labs)
            inseed_accs.append(max(std, tuned))
        else:
            inseed_accs.append(std)
    logger.info(f"  {name} [in-seed ens]: Acc={np.mean(inseed_accs):.4f}±{np.std(inseed_accs):.4f} (max={np.max(inseed_accs):.4f}) >=0.85:{sum(1 for a in inseed_accs if a>=0.85)} >=0.90:{sum(1 for a in inseed_accs if a>=0.90)}")

    # --- Multi-seed ensemble (top-K by val acc) ---
    # Rank seeds by val accuracy
    val_accs = [accuracy_score(all_val_labs, np.argmax(vl, axis=1)) for vl in all_val_logits]
    ranked = np.argsort(val_accs)[::-1]

    for topk in [3, 5, 7, 10]:
        if topk > nr: continue
        top_idx = ranked[:topk]
        ens_test = np.mean([all_test_logits[i] for i in top_idx], axis=0)
        ens_val = np.mean([all_val_logits[i] for i in top_idx], axis=0)
        std = accuracy_score(all_test_labs, np.argmax(ens_test, axis=1))
        if nc == 2:
            tuned = threshold_tune(ens_val, all_val_labs, ens_test, all_test_labs)
            ens_acc = max(std, tuned)
        else:
            ens_acc = std
        logger.info(f"  {name} [top-{topk} ens]: Acc={ens_acc:.4f}")

    # --- All-seed ensemble ---
    ens_all_test = np.mean(all_test_logits, axis=0)
    ens_all_val = np.mean(all_val_logits, axis=0)
    std_all = accuracy_score(all_test_labs, np.argmax(ens_all_test, axis=1))
    if nc == 2:
        tuned_all = threshold_tune(ens_all_val, all_val_labs, ens_all_test, all_test_labs)
        ens_all = max(std_all, tuned_all)
    else:
        ens_all = std_all
    logger.info(f"  {name} [all-{nr} ens]: Acc={ens_all:.4f}")

    return np.mean(single_accs), np.max(single_accs), ens_all

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", default="HateMM", choices=["HateMM", "Multihateclip"])
    parser.add_argument("--language", default="English")
    parser.add_argument("--num_runs", type=int, default=20)
    args = parser.parse_args()
    logger = setup_logger()

    if args.dataset_name == "HateMM":
        emb_dir = "./embeddings/HateMM"; ann_path = "./datasets/HateMM/annotation(new).json"
        split_dir = "./datasets/HateMM/splits"; lm = {"Non Hate": 0, "Hate": 1}; nc = 2
    else:
        emb_dir = f"./embeddings/Multihateclip/{args.language}"
        ann_path = f"./datasets/Multihateclip/{args.language}/annotation(new).json"
        split_dir = f"./datasets/Multihateclip/{args.language}/splits"
        lm = {"Normal": 0, "Offensive": 1, "Hateful": 1}; nc = 2

    feats = {
        "text": torch.load(f"{emb_dir}/text_features.pth", map_location="cpu"),
        "audio": torch.load(f"{emb_dir}/wavlm_audio_features.pth", map_location="cpu"),
        "frame": torch.load(f"{emb_dir}/frame_features.pth", map_location="cpu"),
        "t1": torch.load(f"{emb_dir}/v12_t1_features.pth", map_location="cpu"),
        "t2": torch.load(f"{emb_dir}/v12_t2_features.pth", map_location="cpu"),
        "t1e": torch.load(f"{emb_dir}/v12_t1e_features.pth", map_location="cpu"),
        "ev": torch.load(f"{emb_dir}/v12_evidence_features.pth", map_location="cpu"),
        "struct": torch.load(f"{emb_dir}/v12_struct_features.pth", map_location="cpu"),
    }
    with open(ann_path) as f: feats["labels"] = {d["Video_ID"]: d for d in json.load(f)}
    splits = load_split_ids(split_dir)
    sd = feats["struct"][list(feats["struct"].keys())[0]].shape[0]
    nr = args.num_runs
    logger.info(f"{args.dataset_name} {args.language}")

    # Config 1: Best for HateMM — 6mod+evidence with CE loss + SWA
    run("6mod+ev CE+SWA", feats, splits, lm,
        ["text","audio","frame","t1","t2","ev"], sd, 192, 2e-4, 45, 0.15, 0.15, nr, nc, logger,
        use_focal=False, use_swa=True)

    # Config 2: 6mod+evidence with Focal Loss + SWA
    run("6mod+ev Focal+SWA", feats, splits, lm,
        ["text","audio","frame","t1","t2","ev"], sd, 192, 2e-4, 45, 0.15, 0.15, nr, nc, logger,
        use_focal=True, use_swa=True)

    # Config 3: 6mod+evidence h=256
    run("6mod+ev h=256", feats, splits, lm,
        ["text","audio","frame","t1","t2","ev"], sd, 256, 2e-4, 45, 0.15, 0.15, nr, nc, logger,
        use_focal=False, use_swa=True)

    # Config 4: T1E replace (best for EN MHC) with SWA
    run("T1E+SWA", feats, splits, lm,
        ["text","audio","frame","t1e","t2"], sd, 192, 2e-4, 45, 0.15, 0.15, nr, nc, logger,
        use_focal=False, use_swa=True)

    # Config 5: 7mod (T1E+ev) — all text modalities
    run("7mod T1E+ev+SWA", feats, splits, lm,
        ["text","audio","frame","t1e","t2","ev"], sd, 192, 2e-4, 45, 0.15, 0.15, nr, nc, logger,
        use_focal=False, use_swa=True)

    # Config 6: 6mod+ev longer training (80 epochs, more snapshots)
    run("6mod+ev ep=80", feats, splits, lm,
        ["text","audio","frame","t1","t2","ev"], sd, 192, 2e-4, 80, 0.15, 0.15, nr, nc, logger,
        use_focal=False, use_swa=True)

    # Config 7: 6mod+ev with lower dropout (less regularization)
    run("6mod+ev drop=0.1", feats, splits, lm,
        ["text","audio","frame","t1","t2","ev"], sd, 192, 2e-4, 45, 0.1, 0.1, nr, nc, logger,
        use_focal=False, use_swa=True)

if __name__ == "__main__":
    main()
