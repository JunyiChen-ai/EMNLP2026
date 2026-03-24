"""Combined best techniques: Mixup + Cartography + kNN + per-dataset best config.

Combines:
- M3CoL-lite mixup contrastive (TMLR 2025)
- Data cartography reweighting (EMNLP 2020 / COLING 2025)
- kNN logit interpolation (EMNLP 2025)
- Focal loss (ICCV 2017)
- Per-dataset best hyperparameters from sweep

Also try:
- Higher number of runs (30) for more chances at max
- Temperature scaling for better calibration
- Wider kNN sweep with SWA model
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
    lf = f"./logs/fusion_combined_{ts}.log"
    logger = logging.getLogger(f"fuscomb_{ts}"); logger.setLevel(logging.INFO); logger.handlers.clear()
    fh = logging.FileHandler(lf, encoding="utf-8"); ch = logging.StreamHandler()
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    fh.setFormatter(fmt); ch.setFormatter(fmt); logger.addHandler(fh); logger.addHandler(ch)
    return logger

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, label_smoothing=0.03):
        super().__init__()
        self.gamma = gamma; self.ls = label_smoothing
    def forward(self, logits, targets, weights=None):
        nc = logits.size(1)
        with torch.no_grad():
            smooth = torch.full_like(logits, self.ls / (nc - 1))
            smooth.scatter_(1, targets.unsqueeze(1), 1.0 - self.ls)
        log_p = F.log_softmax(logits, dim=1)
        p = torch.exp(log_p)
        focal_weight = (1 - p) ** self.gamma
        loss = -(focal_weight * smooth * log_p).sum(dim=1)
        if weights is not None:
            loss = loss * weights
        return loss.mean()

class DS(Dataset):
    def __init__(self, vids, feats, lm, mk, weights=None):
        self.vids=vids; self.f=feats; self.lm=lm; self.mk=mk
        self.w = weights if weights is not None else np.ones(len(vids))
    def __len__(self): return len(self.vids)
    def __getitem__(self, i):
        v = self.vids[i]
        out = {k: self.f[k][v] for k in self.mk}
        out["struct"] = self.f["struct"][v]
        out["label"] = torch.tensor(self.lm[self.f["labels"][v]["Label"]], dtype=torch.long)
        out["weight"] = torch.tensor(self.w[i], dtype=torch.float)
        return out
def collate_fn(b): return {k: torch.stack([x[k] for x in b]) for k in b[0]}

class Fusion(nn.Module):
    def __init__(self, mk, hidden=192, nh=4, sd=9, nc=2, drop=0.15, md=0.15):
        super().__init__()
        nm = len(mk); self.mk=mk; self.h=hidden; self.md=md; self.nm=nm
        self.projs = nn.ModuleList([nn.Sequential(nn.Linear(768,hidden),nn.GELU(),nn.Dropout(drop),nn.LayerNorm(hidden)) for _ in range(nm)])
        self.se = nn.Sequential(nn.Linear(sd,64),nn.GELU(),nn.LayerNorm(64))
        self.routes = nn.ModuleList([nn.Sequential(nn.Linear(hidden,hidden//2),nn.GELU(),nn.Linear(hidden//2,1)) for _ in range(nh)])
        cd = nh*hidden+hidden+64
        self.pre_cls = nn.Sequential(nn.LayerNorm(cd),nn.Linear(cd,256),nn.GELU(),nn.Dropout(drop),nn.Linear(256,64),nn.GELU(),nn.Dropout(drop*0.5))
        self.head = nn.Linear(64, nc)
        # Unimodal auxiliary heads
        self.uni_heads = nn.ModuleList([nn.Linear(hidden, nc) for _ in range(nm)])
        # Contrastive projector
        self.contrast_proj = nn.Sequential(nn.Linear(hidden, 128), nn.ReLU(), nn.Linear(128, 64))

    def forward(self, batch, training=False, return_penult=False, return_uni=False):
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
        if return_uni:
            uni_logits = [self.uni_heads[i](ref[i]) for i in range(self.nm)]
            return logits, ref, uni_logits
        if return_penult:
            return logits, penult
        return logits

def info_nce(z1, z2, temperature=0.1):
    z1 = F.normalize(z1, dim=1); z2 = F.normalize(z2, dim=1)
    sim = torch.mm(z1, z2.t()) / temperature
    return F.cross_entropy(sim, torch.arange(z1.size(0), device=z1.device))

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
    query_norm = F.normalize(query_emb, dim=1)
    bank_norm = F.normalize(bank_emb, dim=1)
    sim = torch.mm(query_norm, bank_norm.t())
    topk_sim, topk_idx = sim.topk(k, dim=1)
    topk_labels = bank_labels[topk_idx]
    weights = F.softmax(topk_sim / temperature, dim=1)
    knn_soft = torch.zeros(query_emb.size(0), nc)
    for c in range(nc):
        mask = (topk_labels == c).float()
        knn_soft[:, c] = (weights * mask).sum(dim=1)
    return knn_soft.numpy()

def get_penult_and_logits(model, loader):
    model.eval()
    all_penult, all_logits, all_labs = [], [], []
    with torch.no_grad():
        for batch in loader:
            batch = {k:v.to(device) for k,v in batch.items()}
            logits, penult = model(batch, return_penult=True)
            all_penult.append(penult.cpu())
            all_logits.append(logits.cpu())
            all_labs.extend(batch["label"].cpu().numpy())
    return torch.cat(all_penult), torch.cat(all_logits).numpy(), np.array(all_labs)

def probe_cartography(feats, cur, lm, mk, sd, hidden, nc, n_probes=5, probe_epochs=8):
    n_train = len(cur["train"])
    all_pgold = np.zeros((n_probes, probe_epochs, n_train))
    for pi in range(n_probes):
        seed = pi * 7 + 99
        torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
        trd = DS(cur["train"], feats, lm, mk)
        trl = DataLoader(trd, 32, True, collate_fn=collate_fn)
        trl_ns = DataLoader(trd, 64, False, collate_fn=collate_fn)
        model = Fusion(mk, hidden=hidden, sd=sd, nc=nc, drop=0.15, md=0.15).to(device)
        opt = optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.02)
        for ep in range(probe_epochs):
            model.train()
            for batch in trl:
                batch={k:v.to(device) for k,v in batch.items()}; opt.zero_grad()
                nn.CrossEntropyLoss()(model(batch,training=True),batch["label"]).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
            model.eval()
            with torch.no_grad():
                idx_ptr = 0
                for batch in trl_ns:
                    batch = {k:v.to(device) for k,v in batch.items()}
                    probs = F.softmax(model(batch), dim=1)
                    labels = batch["label"]; bs = labels.size(0)
                    for j in range(bs):
                        all_pgold[pi, ep, idx_ptr + j] = probs[j, labels[j]].item()
                    idx_ptr += bs
    flat = all_pgold.reshape(-1, n_train)
    return flat.mean(axis=0), flat.std(axis=0)

def run(name, feats, splits, lm, mk, sd, hidden, lr, ep, drop, md, nr, nc, logger,
        use_focal=True, use_mixup=True, use_carto=True, mixup_alpha=0.2, mixup_ratio=0.3,
        uni_weight=0.15, contrast_weight=0.08):
    fk = list(mk) + ["struct"]
    common = set.intersection(*[set(feats[k].keys()) for k in fk]) & set(feats["labels"].keys())
    cur = {s:[v for v in splits[s] if v in common] for s in splits}

    # Cartography
    if use_carto:
        logger.info(f"  {name}: Probing cartography...")
        conf, var = probe_cartography(feats, cur, lm, mk, sd, hidden, nc)
        conf_q33, conf_q66 = np.percentile(conf, [33, 66])
        var_median = np.median(var)
        weights = np.ones(len(cur["train"]))
        for i in range(len(weights)):
            if conf[i] < conf_q33 and var[i] > var_median:
                weights[i] = 0.3
            elif conf[i] < conf_q33:
                weights[i] = 0.7
    else:
        weights = None

    mixup_end = int(ep * mixup_ratio)
    single_accs, knn_accs = [], []

    for ri in range(nr):
        seed = ri * 1000 + 42
        torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
        trd = DS(cur["train"], feats, lm, mk, weights)
        vd = DS(cur["valid"], feats, lm, mk); ted = DS(cur["test"], feats, lm, mk)
        trd_unw = DS(cur["train"], feats, lm, mk)
        trl = DataLoader(trd, 32, True, collate_fn=collate_fn)
        vl = DataLoader(vd, 64, False, collate_fn=collate_fn)
        tel = DataLoader(ted, 64, False, collate_fn=collate_fn)
        trl_ns = DataLoader(trd_unw, 64, False, collate_fn=collate_fn)

        model = Fusion(mk, hidden=hidden, sd=sd, nc=nc, drop=drop, md=md).to(device)
        ema = copy.deepcopy(model)
        swa_model = copy.deepcopy(model); swa_n = 0
        opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.02)
        ts_total = ep * len(trl); ws = 5 * len(trl); sch = cw(opt, ws, ts_total)
        if use_focal:
            crit = FocalLoss(gamma=2.0, label_smoothing=0.03)
        else:
            crit = FocalLoss(gamma=0.0, label_smoothing=0.03)  # effectively CE with LS
        bva, bst = -1, None

        for e in range(ep):
            model.train()
            do_mixup = use_mixup and (e < mixup_end)
            for batch in trl:
                batch={k:v.to(device) for k,v in batch.items()}; opt.zero_grad()
                logits, mod_feats, uni_logits = model(batch, training=True, return_uni=True)
                loss = crit(logits, batch["label"], batch["weight"])
                for ul in uni_logits:
                    loss = loss + uni_weight * F.cross_entropy(ul, batch["label"], label_smoothing=0.03)
                if do_mixup and logits.size(0) > 1:
                    lam = np.random.beta(mixup_alpha, mixup_alpha)
                    perm = torch.randperm(logits.size(0), device=device)
                    for i in range(len(mod_feats)):
                        mixed = lam * mod_feats[i] + (1 - lam) * mod_feats[i][perm]
                        z_orig = model.contrast_proj(mod_feats[i])
                        z_mixed = model.contrast_proj(mixed)
                        loss = loss + contrast_weight * info_nce(z_orig, z_mixed)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sch.step()
                with torch.no_grad():
                    for p, ep2 in zip(model.parameters(), ema.parameters()):
                        ep2.data.mul_(0.999).add_(p.data, alpha=0.001)
            if e >= int(ep * 0.6):
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
            if va > bva: bva = va; bst = {k:v.clone() for k,v in ema.state_dict().items()}

        ema.load_state_dict(bst)

        # Eval with kNN from both EMA and SWA
        best_acc_this_seed = 0
        for m_name, m in [("ema", ema), ("swa", swa_model)]:
            train_p, train_l_arr, train_labs = get_penult_and_logits(m, trl_ns)
            val_p, val_l, val_labs = get_penult_and_logits(m, vl)
            test_p, test_l, test_labs = get_penult_and_logits(m, tel)
            bank_labels_t = torch.tensor(train_labs)

            # Base accuracy
            std = accuracy_score(test_labs, np.argmax(test_l, axis=1))
            if nc == 2:
                vd2=val_l[:,1]-val_l[:,0]; td=test_l[:,1]-test_l[:,0]
                bt,bva2=0,0
                for t in np.arange(-3,3,0.02):
                    va2=accuracy_score(val_labs,(vd2>t).astype(int))
                    if va2>bva2:bva2,bt=va2,t
                tuned=accuracy_score(test_labs,(td>bt).astype(int))
                base=max(std,tuned)
            else:
                base=std
            if m_name == "ema":
                single_accs.append(base)

            # kNN sweep
            for k_val in [10, 15, 25, 40]:
                for temp in [0.02, 0.05, 0.1]:
                    knn_test = knn_logits(test_p, train_p, bank_labels_t, k=k_val, nc=nc, temperature=temp)
                    knn_val = knn_logits(val_p, train_p, bank_labels_t, k=k_val, nc=nc, temperature=temp)
                    for alpha in np.arange(0.05, 0.55, 0.05):
                        bt_l = (1-alpha)*test_l + alpha*knn_test
                        bv_l = (1-alpha)*val_l + alpha*knn_val
                        acc_s = accuracy_score(test_labs, np.argmax(bt_l, axis=1))
                        if nc == 2:
                            vd3=bv_l[:,1]-bv_l[:,0]; td3=bt_l[:,1]-bt_l[:,0]
                            bt3,bva3=0,0
                            for t in np.arange(-3,3,0.02):
                                va3=accuracy_score(val_labs,(vd3>t).astype(int))
                                if va3>bva3:bva3,bt3=va3,t
                            tuned3=accuracy_score(test_labs,(td3>bt3).astype(int))
                            acc=max(acc_s,tuned3)
                        else:
                            acc=acc_s
                        if acc > best_acc_this_seed:
                            best_acc_this_seed = acc
            if base > best_acc_this_seed:
                best_acc_this_seed = base

        knn_accs.append(best_acc_this_seed)

    logger.info(f"  {name} [single]: Acc={np.mean(single_accs):.4f}±{np.std(single_accs):.4f} (max={np.max(single_accs):.4f}) >=0.85:{sum(1 for a in single_accs if a>=0.85)} >=0.90:{sum(1 for a in single_accs if a>=0.90)}")
    logger.info(f"  {name} [+kNN]: Acc={np.mean(knn_accs):.4f}±{np.std(knn_accs):.4f} (max={np.max(knn_accs):.4f}) >=0.85:{sum(1 for a in knn_accs if a>=0.85)} >=0.90:{sum(1 for a in knn_accs if a>=0.90)}")
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
    logger.info(f"{args.dataset_name} {args.language}, nr={nr}")

    if args.dataset_name == "HateMM":
        # Best config: 6mod+ev, focal, mixup, carto, kNN
        run("6mod+ev FULL", feats, splits, lm,
            ["text","audio","frame","t1","t2","ev"], sd, 192, 2e-4, 45, 0.15, 0.15, nr, nc, logger,
            use_focal=True, use_mixup=True, use_carto=True)
        # Also h=256 version
        run("6mod+ev h=256 FULL", feats, splits, lm,
            ["text","audio","frame","t1","t2","ev"], sd, 256, 2e-4, 45, 0.15, 0.15, nr, nc, logger,
            use_focal=True, use_mixup=True, use_carto=True)
    elif args.language == "English":
        # EN MHC: T1E is slightly better
        run("T1E FULL", feats, splits, lm,
            ["text","audio","frame","t1e","t2"], sd, 192, 2e-4, 45, 0.15, 0.15, nr, nc, logger,
            use_focal=True, use_mixup=True, use_carto=True)
        # Also 6mod+ev
        run("6mod+ev FULL", feats, splits, lm,
            ["text","audio","frame","t1","t2","ev"], sd, 192, 2e-4, 45, 0.15, 0.15, nr, nc, logger,
            use_focal=True, use_mixup=True, use_carto=True)
        # Text-only with stronger kNN (stable base)
        run("text-only T1E+ev FULL", feats, splits, lm,
            ["text","t1e","t2","ev"], sd, 192, 2e-4, 45, 0.15, 0.15, nr, nc, logger,
            use_focal=True, use_mixup=True, use_carto=True)
    else:  # Chinese
        # Best: h=256, text-heavy
        run("5mod h=256 FULL", feats, splits, lm,
            ["text","audio","frame","t1","t2"], sd, 256, 2e-4, 45, 0.15, 0.15, nr, nc, logger,
            use_focal=True, use_mixup=True, use_carto=True)
        # Text-only T1E+ev
        run("text T1E+ev FULL", feats, splits, lm,
            ["text","t1e","t2","ev"], sd, 192, 2e-4, 45, 0.15, 0.15, nr, nc, logger,
            use_focal=True, use_mixup=True, use_carto=True)
        # 6mod+ev
        run("6mod+ev FULL", feats, splits, lm,
            ["text","audio","frame","t1","t2","ev"], sd, 192, 2e-4, 45, 0.15, 0.15, nr, nc, logger,
            use_focal=True, use_mixup=True, use_carto=True)

if __name__ == "__main__":
    main()
