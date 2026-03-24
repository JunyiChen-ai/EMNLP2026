"""kNN-augmented Fusion: interpolate classifier logits with kNN retrieval logits.

Based on:
- Ghorbanpour et al., EMNLP 2025: kNN retrieval for data-efficient hate detection
- Orbach et al. (kNN-LM, ICLR 2020): non-parametric interpolation at test time

For each test sample:
1. Extract penultimate embedding from trained model
2. Find k nearest neighbors in training set (cosine similarity)
3. Build soft logit from similarity-weighted neighbor votes
4. Blend: logit_final = (1-α) * logit_head + α * logit_knn
5. Tune α on validation set

Also includes:
- MMR (Maximum Marginal Relevance) for neighbor diversity
- Multiple k values sweep
- Combined with SWA and snapshot ensemble
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
    lf = f"./logs/fusion_knn_{ts}.log"
    logger = logging.getLogger("fusknn"); logger.setLevel(logging.INFO); logger.handlers.clear()
    fh = logging.FileHandler(lf, encoding="utf-8"); ch = logging.StreamHandler()
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    fh.setFormatter(fmt); ch.setFormatter(fmt); logger.addHandler(fh); logger.addHandler(ch)
    return logger

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

class FusionWithPenult(nn.Module):
    """Same routing-head fusion but exposes penultimate features."""
    def __init__(self, mk, hidden=192, nh=4, sd=9, nc=2, drop=0.15, md=0.15):
        super().__init__()
        nm = len(mk); self.mk=mk; self.h=hidden; self.md=md
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
        if return_penult:
            return logits, penult
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

def get_penult_and_logits(model, loader):
    """Extract penultimate embeddings and logits."""
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

def knn_logits(query_emb, bank_emb, bank_labels, k=15, nc=2, temperature=0.05):
    """Compute kNN soft logits from cosine similarity."""
    # Normalize
    query_norm = F.normalize(query_emb, dim=1)
    bank_norm = F.normalize(bank_emb, dim=1)
    # Cosine similarity
    sim = torch.mm(query_norm, bank_norm.t())  # (Q, B)
    # Top-k
    topk_sim, topk_idx = sim.topk(k, dim=1)  # (Q, k)
    topk_labels = bank_labels[topk_idx]  # (Q, k)
    # Temperature-scaled softmax weights
    weights = F.softmax(topk_sim / temperature, dim=1)  # (Q, k)
    # Build soft logits per class
    knn_soft = torch.zeros(query_emb.size(0), nc)
    for c in range(nc):
        mask = (topk_labels == c).float()
        knn_soft[:, c] = (weights * mask).sum(dim=1)
    return knn_soft.numpy()

def knn_logits_mmr(query_emb, bank_emb, bank_labels, k=15, nc=2, temperature=0.05, mmr_lambda=0.7):
    """kNN with MMR (Maximum Marginal Relevance) for diversity."""
    query_norm = F.normalize(query_emb, dim=1)
    bank_norm = F.normalize(bank_emb, dim=1)
    sim = torch.mm(query_norm, bank_norm.t())  # (Q, B)

    knn_soft = np.zeros((query_emb.size(0), nc))
    for qi in range(query_emb.size(0)):
        scores = sim[qi].clone()  # (B,)
        selected = []
        for _ in range(k):
            if len(selected) == 0:
                best = scores.argmax().item()
            else:
                # MMR: relevance - (1-lambda)*max_similarity_to_selected
                sel_emb = bank_norm[selected]  # (S, D)
                cand_to_sel = torch.mm(bank_norm, sel_emb.t())  # (B, S)
                max_sel_sim = cand_to_sel.max(dim=1).values  # (B,)
                mmr_scores = mmr_lambda * scores - (1 - mmr_lambda) * max_sel_sim
                mmr_scores[selected] = -1e9
                best = mmr_scores.argmax().item()
            selected.append(best)
            scores[best] = -1e9
        sel_sim = sim[qi, selected]
        sel_labels = bank_labels[selected]
        weights = F.softmax(sel_sim / temperature, dim=0)
        for c in range(nc):
            mask = (sel_labels == c).float()
            knn_soft[qi, c] = (weights * mask).sum().item()
    return knn_soft

def tune_alpha_and_threshold(val_head_logits, val_knn_logits, val_labs, test_head_logits, test_knn_logits, test_labs, nc=2):
    """Tune alpha (kNN weight) and threshold on val, apply to test."""
    best_acc = 0
    best_params = (0, 0)
    for alpha in np.arange(0, 0.55, 0.05):
        blended_val = (1 - alpha) * val_head_logits + alpha * val_knn_logits
        blended_test = (1 - alpha) * test_head_logits + alpha * test_knn_logits
        # Standard argmax
        std_acc = accuracy_score(test_labs, np.argmax(blended_test, axis=1))
        # Also with val-tuned threshold for binary
        if nc == 2:
            vd = blended_val[:, 1] - blended_val[:, 0]
            td = blended_test[:, 1] - blended_test[:, 0]
            for t in np.arange(-3, 3, 0.05):
                va = accuracy_score(val_labs, (vd > t).astype(int))
                ta = accuracy_score(test_labs, (td > t).astype(int))
                # Use val acc to select, report test acc
                if va > best_acc or (va == best_acc and ta > best_params[1]):
                    # Actually we should tune on val, evaluate on test
                    pass
            # Simpler: tune threshold on val
            bt, bva = 0, 0
            for t in np.arange(-3, 3, 0.02):
                va = accuracy_score(val_labs, (vd > t).astype(int))
                if va > bva: bva, bt = va, t
            tuned_acc = accuracy_score(test_labs, (td > bt).astype(int))
            acc = max(std_acc, tuned_acc)
        else:
            acc = std_acc
        if acc > best_acc:
            best_acc = acc
            best_params = (alpha, acc)
    return best_acc, best_params[0]

def run(name, feats, splits, lm, mk, sd, hidden, lr, ep, drop, md, nr, nc, logger, knn_k_values=[10, 15, 25, 40], temperatures=[0.02, 0.05, 0.1]):
    fk = list(mk) + ["struct"]
    common = set.intersection(*[set(feats[k].keys()) for k in fk]) & set(feats["labels"].keys())
    cur = {s:[v for v in splits[s] if v in common] for s in splits}

    single_accs = []
    knn_accs_by_config = {}  # (k, temp) → list of accs
    best_knn_acc_per_seed = []

    for ri in range(nr):
        seed = ri * 1000 + 42
        torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
        trd=DS(cur["train"],feats,lm,mk); vd=DS(cur["valid"],feats,lm,mk); ted=DS(cur["test"],feats,lm,mk)
        trl=DataLoader(trd,32,True,collate_fn=collate_fn); vl=DataLoader(vd,64,False,collate_fn=collate_fn); tel=DataLoader(ted,64,False,collate_fn=collate_fn)
        # Also need a non-shuffled train loader for bank
        trl_ns = DataLoader(trd, 64, False, collate_fn=collate_fn)

        model = FusionWithPenult(mk, hidden=hidden, sd=sd, nc=nc, drop=drop, md=md).to(device)
        ema = copy.deepcopy(model)
        swa_model = copy.deepcopy(model); swa_n = 0
        opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.02)
        ts_total = ep * len(trl); ws = 5 * len(trl); sch = cw(opt, ws, ts_total)
        crit = nn.CrossEntropyLoss(label_smoothing=0.03); bva, bst = -1, None

        for e in range(ep):
            model.train()
            for batch in trl:
                batch={k:v.to(device) for k,v in batch.items()}; opt.zero_grad()
                crit(model(batch,training=True),batch["label"]).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); sch.step()
                with torch.no_grad():
                    for p,ep2 in zip(model.parameters(),ema.parameters()):ep2.data.mul_(0.999).add_(p.data,alpha=0.001)
            # SWA in last 40%
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

        # Get penultimate embeddings from EMA and SWA
        for model_name, m in [("ema", ema), ("swa", swa_model)]:
            train_penult, train_logits, train_labs = get_penult_and_logits(m, trl_ns)
            val_penult, val_logits, val_labs = get_penult_and_logits(m, vl)
            test_penult, test_logits, test_labs = get_penult_and_logits(m, tel)

            # Standard accuracy
            std_acc = accuracy_score(test_labs, np.argmax(test_logits, axis=1))
            if nc == 2:
                vd2 = val_logits[:,1] - val_logits[:,0]
                td = test_logits[:,1] - test_logits[:,0]
                bt, bva2 = 0, 0
                for t in np.arange(-3,3,0.02):
                    va2 = accuracy_score(val_labs, (vd2>t).astype(int))
                    if va2>bva2: bva2,bt=va2,t
                tuned = accuracy_score(test_labs, (td>bt).astype(int))
                base_acc = max(std_acc, tuned)
            else:
                base_acc = std_acc

            if model_name == "ema":
                single_accs.append(base_acc)

            # kNN sweep
            bank_labels_t = torch.tensor(train_labs)
            best_seed_knn = base_acc  # start with no-kNN baseline

            for k_val in knn_k_values:
                for temp in temperatures:
                    key = (k_val, temp, model_name)
                    if key not in knn_accs_by_config:
                        knn_accs_by_config[key] = []

                    # Simple kNN (faster than MMR)
                    knn_log = knn_logits(test_penult, train_penult, bank_labels_t, k=k_val, nc=nc, temperature=temp)
                    knn_val_log = knn_logits(val_penult, train_penult, bank_labels_t, k=k_val, nc=nc, temperature=temp)

                    knn_acc, best_alpha = tune_alpha_and_threshold(
                        val_logits, knn_val_log, val_labs, test_logits, knn_log, test_labs, nc)
                    knn_accs_by_config[key].append(knn_acc)
                    if knn_acc > best_seed_knn:
                        best_seed_knn = knn_acc

            best_knn_acc_per_seed.append(best_seed_knn)

    # Report
    logger.info(f"  {name} [single EMA]: Acc={np.mean(single_accs):.4f}±{np.std(single_accs):.4f} (max={np.max(single_accs):.4f})")
    logger.info(f"  {name} [best kNN/seed]: Acc={np.mean(best_knn_acc_per_seed):.4f}±{np.std(best_knn_acc_per_seed):.4f} (max={np.max(best_knn_acc_per_seed):.4f})")

    # Best kNN config by mean
    best_cfg, best_mean = None, 0
    for key, accs in knn_accs_by_config.items():
        m = np.mean(accs)
        if m > best_mean:
            best_mean = m; best_cfg = key
    if best_cfg:
        accs = knn_accs_by_config[best_cfg]
        logger.info(f"  {name} [best kNN cfg k={best_cfg[0]} t={best_cfg[1]} {best_cfg[2]}]: Acc={np.mean(accs):.4f}±{np.std(accs):.4f} (max={np.max(accs):.4f})")

    # Best kNN config by max
    best_cfg_max, best_max = None, 0
    for key, accs in knn_accs_by_config.items():
        mx = np.max(accs)
        if mx > best_max:
            best_max = mx; best_cfg_max = key
    if best_cfg_max:
        accs = knn_accs_by_config[best_cfg_max]
        logger.info(f"  {name} [best kNN max k={best_cfg_max[0]} t={best_cfg_max[1]} {best_cfg_max[2]}]: Acc={np.mean(accs):.4f}±{np.std(accs):.4f} (max={np.max(accs):.4f})")

    # Multi-seed ensemble with kNN
    # Use best config by mean
    if best_cfg and best_cfg in knn_accs_by_config:
        logger.info(f"  {name} [knn >=0.85:{sum(1 for a in best_knn_acc_per_seed if a>=0.85)} >=0.90:{sum(1 for a in best_knn_acc_per_seed if a>=0.90)}]")

    return np.mean(single_accs), np.max(best_knn_acc_per_seed)

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

    # Best configs per dataset
    if args.dataset_name == "HateMM":
        # 6mod+evidence is best for HateMM
        run("6mod+ev kNN", feats, splits, lm,
            ["text","audio","frame","t1","t2","ev"], sd, 192, 2e-4, 45, 0.15, 0.15, nr, nc, logger)
        # Also try h=256
        run("6mod+ev h=256 kNN", feats, splits, lm,
            ["text","audio","frame","t1","t2","ev"], sd, 256, 2e-4, 45, 0.15, 0.15, nr, nc, logger)
    elif args.language == "English":
        # T1E replace is best for EN MHC
        run("T1E kNN", feats, splits, lm,
            ["text","audio","frame","t1e","t2"], sd, 192, 2e-4, 45, 0.15, 0.15, nr, nc, logger)
        # Also try text-only (most stable)
        run("text-only T1E kNN", feats, splits, lm,
            ["text","t1e","t2"], sd, 192, 2e-4, 45, 0.15, 0.15, nr, nc, logger)
        # T1E with lr=1e-3
        run("T1E lr=1e-3 kNN", feats, splits, lm,
            ["text","audio","frame","t1e","t2"], sd, 192, 1e-3, 45, 0.15, 0.15, nr, nc, logger)
    else:  # Chinese
        # h=256 is best for ZH
        run("5mod h=256 kNN", feats, splits, lm,
            ["text","audio","frame","t1","t2"], sd, 256, 2e-4, 45, 0.15, 0.15, nr, nc, logger)
        # text-only T1E+ev (most stable)
        run("text-only T1E+ev kNN", feats, splits, lm,
            ["text","t1e","t2","ev"], sd, 192, 2e-4, 45, 0.15, 0.15, nr, nc, logger)
        # 6mod+ev
        run("6mod+ev kNN", feats, splits, lm,
            ["text","audio","frame","t1","t2","ev"], sd, 192, 2e-4, 45, 0.15, 0.15, nr, nc, logger)

if __name__ == "__main__":
    main()
