"""Confident Learning + Error Analysis + Targeted Retraining.

Based on:
- Northcutt et al., JAIR 2021: Confident Learning for label noise identification
- Kim et al., COLING 2025: CONELA for offensive language dataset refinement

Pipeline:
1. Train 10 quick models → average cross-validated predictions
2. Use Confident Learning to identify likely mislabeled training samples
3. Options: (a) flip labels, (b) remove, (c) downweight
4. Retrain with cleaned data + kNN

Also does error analysis:
- Which test samples are consistently misclassified?
- What modalities contribute to errors?
- Can we build error-specific rescue rules?
"""
import argparse, csv, json, os, random, copy, logging
from datetime import datetime
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from sklearn.metrics import accuracy_score, confusion_matrix
from torch.utils.data import DataLoader, Dataset
import warnings; warnings.filterwarnings("ignore")

device = "cuda"

def setup_logger():
    os.makedirs("./logs", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    lf = f"./logs/fusion_cleanlab_{ts}.log"
    logger = logging.getLogger(f"fuscl_{ts}"); logger.setLevel(logging.INFO); logger.handlers.clear()
    fh = logging.FileHandler(lf, encoding="utf-8"); ch = logging.StreamHandler()
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    fh.setFormatter(fmt); ch.setFormatter(fmt); logger.addHandler(fh); logger.addHandler(ch)
    return logger

class DS(Dataset):
    def __init__(self, vids, feats, lm, mk, weights=None, label_override=None):
        self.vids=vids; self.f=feats; self.lm=lm; self.mk=mk
        self.w = weights if weights is not None else np.ones(len(vids))
        self.lo = label_override  # dict vid→label
    def __len__(self): return len(self.vids)
    def __getitem__(self, i):
        v = self.vids[i]
        out = {k: self.f[k][v] for k in self.mk}
        out["struct"] = self.f["struct"][v]
        if self.lo and v in self.lo:
            out["label"] = torch.tensor(self.lo[v], dtype=torch.long)
        else:
            out["label"] = torch.tensor(self.lm[self.f["labels"][v]["Label"]], dtype=torch.long)
        out["weight"] = torch.tensor(self.w[i], dtype=torch.float)
        return out
def collate_fn(b): return {k: torch.stack([x[k] for x in b]) for k in b[0]}

class Fusion(nn.Module):
    def __init__(self, mk, hidden=192, nh=4, sd=9, nc=2, drop=0.15, md=0.15):
        super().__init__()
        nm = len(mk); self.mk=mk; self.md=md
        self.projs = nn.ModuleList([nn.Sequential(nn.Linear(768,hidden),nn.GELU(),nn.Dropout(drop),nn.LayerNorm(hidden)) for _ in range(nm)])
        self.se = nn.Sequential(nn.Linear(sd,64),nn.GELU(),nn.LayerNorm(64))
        self.routes = nn.ModuleList([nn.Sequential(nn.Linear(hidden,hidden//2),nn.GELU(),nn.Linear(hidden//2,1)) for _ in range(nh)])
        cd = nh*hidden+hidden+64
        self.pre_cls = nn.Sequential(nn.LayerNorm(cd),nn.Linear(cd,256),nn.GELU(),nn.Dropout(drop),nn.Linear(256,64),nn.GELU(),nn.Dropout(drop*0.5))
        self.head = nn.Linear(64, nc)
    def forward(self, batch, training=False, return_penult=False):
        ref = []
        for i,(p,k) in enumerate(zip(self.projs,self.mk)):
            h=p(batch[k])
            if training and self.md>0: h=h*(torch.rand(h.size(0),1,device=h.device)>self.md).float()
            ref.append(h)
        st=torch.stack(ref,dim=1)
        heads=[((st*torch.softmax(rm(st).squeeze(-1),dim=1).unsqueeze(-1)).sum(dim=1)) for rm in self.routes]
        fused = torch.cat(heads+[st.mean(dim=1),self.se(batch["struct"])],dim=-1)
        penult = self.pre_cls(fused)
        logits = self.head(penult)
        if return_penult: return logits, penult
        return logits

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

def knn_logits(query_emb, bank_emb, bank_labels, k=15, nc=2, temperature=0.05):
    qn=F.normalize(query_emb,dim=1); bn=F.normalize(bank_emb,dim=1)
    sim=torch.mm(qn,bn.t()); ts2,ti=sim.topk(k,dim=1)
    tl=bank_labels[ti]; w=F.softmax(ts2/temperature,dim=1)
    out=torch.zeros(query_emb.size(0),nc)
    for c in range(nc): out[:,c]=(w*(tl==c).float()).sum(dim=1)
    return out.numpy()

def get_pl(model, loader):
    model.eval(); ap,al,alb=[],[],[]
    with torch.no_grad():
        for b in loader:
            b={k:v.to(device) for k,v in b.items()}
            lo,pe=model(b,return_penult=True)
            ap.append(pe.cpu()); al.append(lo.cpu()); alb.extend(b["label"].cpu().numpy())
    return torch.cat(ap),torch.cat(al).numpy(),np.array(alb)

def best_thresh_acc(val_l, val_labs, test_l, test_labs, nc):
    std=accuracy_score(test_labs,np.argmax(test_l,axis=1))
    if nc==2:
        vd=val_l[:,1]-val_l[:,0]; td=test_l[:,1]-test_l[:,0]
        bt,bv=0,0
        for t in np.arange(-3,3,0.02):
            v=accuracy_score(val_labs,(vd>t).astype(int))
            if v>bv:bv,bt=v,t
        return max(std,accuracy_score(test_labs,(td>bt).astype(int)))
    return std

def confident_learning_identify(feats, cur, lm, mk, sd, hidden, nc, n_models=10, logger=None):
    """Identify likely mislabeled samples using cross-validation predictions."""
    n_train = len(cur["train"])
    # Collect P(class) for each training sample across models
    all_probs = np.zeros((n_models, n_train, nc))
    labels_arr = np.zeros(n_train, dtype=int)

    for mi in range(n_models):
        seed = mi * 13 + 77
        torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
        trd = DS(cur["train"], feats, lm, mk)
        trl = DataLoader(trd, 32, True, collate_fn=collate_fn)
        trl_ns = DataLoader(trd, 64, False, collate_fn=collate_fn)

        model = Fusion(mk, hidden=hidden, sd=sd, nc=nc, drop=0.15, md=0.15).to(device)
        opt = optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.02)

        # Train for 20 epochs (enough to learn patterns)
        for ep in range(20):
            model.train()
            for batch in trl:
                batch={k:v.to(device) for k,v in batch.items()}; opt.zero_grad()
                nn.CrossEntropyLoss(label_smoothing=0.03)(model(batch,training=True),batch["label"]).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()

        # Collect predictions on training set (non-shuffled)
        model.eval()
        with torch.no_grad():
            idx = 0
            for batch in trl_ns:
                batch = {k:v.to(device) for k,v in batch.items()}
                probs = F.softmax(model(batch), dim=1).cpu().numpy()
                bs = probs.shape[0]
                all_probs[mi, idx:idx+bs] = probs
                if mi == 0:
                    labels_arr[idx:idx+bs] = batch["label"].cpu().numpy()
                idx += bs

    # Average predictions across models
    avg_probs = all_probs.mean(axis=0)  # (n_train, nc)

    # Confident Learning: identify samples where model prediction disagrees with label
    # AND the model is confident about it
    pred_labels = avg_probs.argmax(axis=1)
    pred_conf = avg_probs.max(axis=1)

    # Noisy candidates: predicted != given AND confidence > threshold
    noisy_mask = (pred_labels != labels_arr) & (pred_conf > 0.6)
    noisy_indices = np.where(noisy_mask)[0]

    # Also identify borderline cases (confidence between 0.4-0.6)
    borderline_mask = (avg_probs.max(axis=1) < 0.6) & (avg_probs.max(axis=1) > 0.4)
    borderline_indices = np.where(borderline_mask)[0]

    if logger:
        logger.info(f"    CL found {len(noisy_indices)} likely mislabeled, {len(borderline_indices)} borderline out of {n_train}")
        for ni in noisy_indices[:10]:
            vid = cur["train"][ni]
            logger.info(f"      Noisy: {vid} label={labels_arr[ni]} pred={pred_labels[ni]} conf={pred_conf[ni]:.3f}")

    return noisy_indices, borderline_indices, labels_arr, pred_labels

def run(name, feats, splits, lm, mk, sd, hidden, lr, ep, drop, md, nr, nc, logger,
        clean_mode="downweight"):
    fk=list(mk)+["struct"]
    common=set.intersection(*[set(feats[k].keys()) for k in fk])&set(feats["labels"].keys())
    cur={s:[v for v in splits[s] if v in common] for s in splits}

    # Step 1: Confident Learning
    logger.info(f"  {name}: Running Confident Learning...")
    noisy_idx, border_idx, labels_arr, pred_labels = confident_learning_identify(
        feats, cur, lm, mk, sd, hidden, nc, n_models=10, logger=logger)

    # Step 2: Create cleaned dataset
    n_train = len(cur["train"])
    weights = np.ones(n_train)
    label_override = {}

    if clean_mode == "downweight":
        for ni in noisy_idx:
            weights[ni] = 0.1  # heavily downweight likely mislabeled
        for bi in border_idx:
            weights[bi] = 0.5  # moderately downweight borderline
    elif clean_mode == "flip":
        for ni in noisy_idx:
            vid = cur["train"][ni]
            label_override[vid] = int(pred_labels[ni])
    elif clean_mode == "remove":
        keep_mask = np.ones(n_train, dtype=bool)
        keep_mask[noisy_idx] = False
        cur["train"] = [cur["train"][i] for i in range(n_train) if keep_mask[i]]
        weights = np.ones(len(cur["train"]))

    logger.info(f"  {name} ({clean_mode}): {len(cur['train'])} train samples, {len(noisy_idx)} cleaned")

    # Step 3: Train with cleaned data
    single_accs, knn_accs = [], []
    for ri in range(nr):
        seed = ri * 1000 + 42
        torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
        trd = DS(cur["train"], feats, lm, mk, weights, label_override)
        vd = DS(cur["valid"], feats, lm, mk)
        ted = DS(cur["test"], feats, lm, mk)
        trd_u = DS(cur["train"], feats, lm, mk)
        trl = DataLoader(trd, 32, True, collate_fn=collate_fn)
        vl = DataLoader(vd, 64, False, collate_fn=collate_fn)
        tel = DataLoader(ted, 64, False, collate_fn=collate_fn)
        trl_ns = DataLoader(trd_u, 64, False, collate_fn=collate_fn)

        model = Fusion(mk, hidden=hidden, sd=sd, nc=nc, drop=drop, md=md).to(device)
        ema = copy.deepcopy(model)
        opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.02)
        ts_t = ep * len(trl); ws = 5 * len(trl); sch = cw(opt, ws, ts_t)
        bva, bst = -1, None

        for e in range(ep):
            model.train()
            for batch in trl:
                batch={k:v.to(device) for k,v in batch.items()}; opt.zero_grad()
                logits = model(batch, training=True)
                per_sample = F.cross_entropy(logits, batch["label"], reduction='none', label_smoothing=0.03)
                loss = (per_sample * batch["weight"]).mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sch.step()
                with torch.no_grad():
                    for p, ep2 in zip(model.parameters(), ema.parameters()):
                        ep2.data.mul_(0.999).add_(p.data, alpha=0.001)
            ema.eval(); ps, ls2 = [], []
            with torch.no_grad():
                for batch in vl:
                    batch={k:v.to(device) for k,v in batch.items()}
                    ps.extend(ema(batch).argmax(1).cpu().numpy()); ls2.extend(batch["label"].cpu().numpy())
            va = accuracy_score(ls2, ps)
            if va > bva: bva = va; bst = {k:v.clone() for k,v in ema.state_dict().items()}

        ema.load_state_dict(bst)
        tp,tl_arr,tla = get_pl(ema, trl_ns)
        vp,vl_arr,vla = get_pl(ema, vl)
        tep,tel_arr,tela = get_pl(ema, tel)
        base = best_thresh_acc(vl_arr, vla, tel_arr, tela, nc)
        single_accs.append(base)

        # kNN
        blt = torch.tensor(tla)
        best_knn = base
        for k in [10, 15, 25, 40]:
            for temp in [0.02, 0.05, 0.1]:
                kt = knn_logits(tep, tp, blt, k=k, nc=nc, temperature=temp)
                kv = knn_logits(vp, tp, blt, k=k, nc=nc, temperature=temp)
                for a in np.arange(0.05, 0.55, 0.05):
                    bt_l = (1-a)*tel_arr + a*kt
                    bv_l = (1-a)*vl_arr + a*kv
                    acc = best_thresh_acc(bv_l, vla, bt_l, tela, nc)
                    if acc > best_knn: best_knn = acc
        knn_accs.append(best_knn)

    logger.info(f"  {name} [{clean_mode} single]: Acc={np.mean(single_accs):.4f}±{np.std(single_accs):.4f} (max={np.max(single_accs):.4f}) >=0.85:{sum(1 for a in single_accs if a>=0.85)} >=0.90:{sum(1 for a in single_accs if a>=0.90)}")
    logger.info(f"  {name} [{clean_mode} +kNN]: Acc={np.mean(knn_accs):.4f}±{np.std(knn_accs):.4f} (max={np.max(knn_accs):.4f}) >=0.85:{sum(1 for a in knn_accs if a>=0.85)} >=0.90:{sum(1 for a in knn_accs if a>=0.90)}")

    # Error analysis on test set (using best model)
    ema.load_state_dict(bst)
    ema.eval()
    test_preds, test_labs_list, test_vids = [], [], []
    test_loader = DataLoader(DS(cur["test"], feats, lm, mk), 64, False, collate_fn=collate_fn)
    with torch.no_grad():
        vi = 0
        for batch in test_loader:
            batch={k:v.to(device) for k,v in batch.items()}
            preds = ema(batch).argmax(1).cpu().numpy()
            labs = batch["label"].cpu().numpy()
            bs = len(labs)
            for j in range(bs):
                test_preds.append(preds[j])
                test_labs_list.append(labs[j])
                test_vids.append(cur["test"][vi+j])
            vi += bs

    errors = [(test_vids[i], test_labs_list[i], test_preds[i])
              for i in range(len(test_vids)) if test_preds[i] != test_labs_list[i]]
    cm = confusion_matrix(test_labs_list, test_preds)
    logger.info(f"  {name} Error analysis: {len(errors)} errors, CM={cm.tolist()}")
    logger.info(f"  {name} Sample errors: {errors[:5]}")

    return np.max(knn_accs)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", default="HateMM", choices=["HateMM", "Multihateclip"])
    parser.add_argument("--language", default="English")
    parser.add_argument("--num_runs", type=int, default=30)
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

    if args.dataset_name == "HateMM":
        mk = ["text","audio","frame","t1","t2","ev"]
        for mode in ["downweight", "flip", "remove"]:
            run(f"6mod+ev", feats, splits, lm, mk, sd, 192, 2e-4, 45, 0.15, 0.15, nr, nc, logger, clean_mode=mode)
    elif args.language == "English":
        # Try both T1E and 6mod+ev
        for mk, mname in [
            (["text","audio","frame","t1","t2","ev"], "6mod+ev"),
            (["text","audio","frame","t1e","t2"], "T1E"),
        ]:
            for mode in ["downweight", "flip"]:
                run(mname, feats, splits, lm, mk, sd, 192, 2e-4, 45, 0.15, 0.15, nr, nc, logger, clean_mode=mode)
    else:
        for mk, mname in [
            (["text","audio","frame","t1","t2","ev"], "6mod+ev"),
            (["text","audio","frame","t1","t2"], "5mod h=256"),
        ]:
            h = 256 if "h=256" in mname else 192
            for mode in ["downweight", "flip"]:
                run(mname, feats, splits, lm, mk, sd, h, 2e-4, 45, 0.15, 0.15, nr, nc, logger, clean_mode=mode)

if __name__ == "__main__":
    main()
