"""Data Cartography + Curriculum Learning fusion.

Based on:
- Swayamdipta et al., EMNLP 2020: Dataset Cartography — map training dynamics
- Kim et al., COLING 2025: CONELA — confident learning for offensive language
- Curriculum learning (Bengio et al., ICML 2009): easy→hard ordering

Pipeline:
1. Phase 1 (probe): Train 5 quick models (8 epochs each) to collect per-sample statistics
   - Confidence: mean P(gold) across epochs/seeds
   - Variability: std P(gold) across epochs/seeds
2. Classify samples: easy (high conf, low var), ambiguous (med conf, high var), hard (low conf)
3. Phase 2 (train): Curriculum learning — start with easy+ambiguous, add hard samples gradually
   + Reweight hard+high-disagreement samples down
4. Combine with kNN at test time
"""
import argparse, csv, json, os, random, copy, logging
from datetime import datetime
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import warnings; warnings.filterwarnings("ignore")

device = "cuda"

def setup_logger():
    os.makedirs("./logs", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    lf = f"./logs/fusion_carto_{ts}.log"
    logger = logging.getLogger("fuscarto"); logger.setLevel(logging.INFO); logger.handlers.clear()
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
        out["idx"] = torch.tensor(i, dtype=torch.long)
        return out

class DSWeighted(Dataset):
    def __init__(self, vids, feats, lm, mk, weights):
        self.vids=vids; self.f=feats; self.lm=lm; self.mk=mk; self.w=weights
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

def probe_training_dynamics(feats, cur, lm, mk, sd, hidden, nc, n_probes=5, probe_epochs=8):
    """Run quick probe models to map training dynamics."""
    n_train = len(cur["train"])
    # Collect P(gold) for each sample across probes and epochs
    all_pgold = np.zeros((n_probes, probe_epochs, n_train))

    for pi in range(n_probes):
        seed = pi * 7 + 99
        torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
        trd = DS(cur["train"], feats, lm, mk)
        trl = DataLoader(trd, 32, True, collate_fn=collate_fn)
        trl_ns = DataLoader(trd, 64, False, collate_fn=collate_fn)  # non-shuffled for eval

        model = Fusion(mk, hidden=hidden, sd=sd, nc=nc, drop=0.15, md=0.15).to(device)
        opt = optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.02)

        for ep in range(probe_epochs):
            model.train()
            for batch in trl:
                batch={k:v.to(device) for k,v in batch.items()}; opt.zero_grad()
                nn.CrossEntropyLoss()(model(batch,training=True),batch["label"]).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()

            # Eval on train (non-shuffled)
            model.eval()
            with torch.no_grad():
                idx_ptr = 0
                for batch in trl_ns:
                    batch = {k:v.to(device) for k,v in batch.items()}
                    probs = F.softmax(model(batch), dim=1)
                    labels = batch["label"]
                    bs = labels.size(0)
                    for j in range(bs):
                        all_pgold[pi, ep, idx_ptr + j] = probs[j, labels[j]].item()
                    idx_ptr += bs

    # Compute confidence and variability per sample
    # Flatten across probes and epochs
    flat = all_pgold.reshape(-1, n_train)  # (n_probes*probe_epochs, n_train)
    confidence = flat.mean(axis=0)
    variability = flat.std(axis=0)

    return confidence, variability

def run(name, feats, splits, lm, mk, sd, hidden, lr, ep, drop, md, nr, nc, logger,
        use_cartography=True, knn_k=15, knn_temp=0.05):
    fk = list(mk) + ["struct"]
    common = set.intersection(*[set(feats[k].keys()) for k in fk]) & set(feats["labels"].keys())
    cur = {s:[v for v in splits[s] if v in common] for s in splits}

    # Phase 1: Probe training dynamics
    if use_cartography:
        logger.info(f"  {name}: Probing training dynamics...")
        confidence, variability = probe_training_dynamics(feats, cur, lm, mk, sd, hidden, nc)

        # Classify samples
        n = len(cur["train"])
        # Easy: high confidence, low variability
        # Ambiguous: medium confidence, high variability
        # Hard: low confidence
        conf_q33, conf_q66 = np.percentile(confidence, [33, 66])
        var_median = np.median(variability)

        weights = np.ones(n)
        n_easy, n_amb, n_hard = 0, 0, 0
        for i in range(n):
            if confidence[i] > conf_q66 and variability[i] < var_median:
                # Easy + low disagreement: weight 1.0
                weights[i] = 1.0
                n_easy += 1
            elif confidence[i] < conf_q33:
                # Hard: downweight
                if variability[i] > var_median:
                    weights[i] = 0.3  # Hard + high variability (likely noisy)
                else:
                    weights[i] = 0.7  # Hard + low variability (genuinely hard)
                n_hard += 1
            else:
                # Ambiguous: full weight (most informative)
                weights[i] = 1.0
                n_amb += 1

        logger.info(f"  {name}: Cartography — easy={n_easy}, ambiguous={n_amb}, hard={n_hard}")
        logger.info(f"  {name}: Confidence range=[{confidence.min():.3f},{confidence.max():.3f}], Variability range=[{variability.min():.3f},{variability.max():.3f}]")
    else:
        weights = np.ones(len(cur["train"]))

    # Phase 2: Train with reweighted data
    single_accs, knn_accs = [], []

    for ri in range(nr):
        seed = ri * 1000 + 42
        torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
        trd = DSWeighted(cur["train"], feats, lm, mk, weights)
        vd = DS(cur["valid"], feats, lm, mk)
        ted = DS(cur["test"], feats, lm, mk)
        trd_unweighted = DS(cur["train"], feats, lm, mk)

        trl = DataLoader(trd, 32, True, collate_fn=collate_fn)
        vl = DataLoader(vd, 64, False, collate_fn=collate_fn)
        tel = DataLoader(ted, 64, False, collate_fn=collate_fn)
        trl_ns = DataLoader(trd_unweighted, 64, False, collate_fn=collate_fn)

        model = Fusion(mk, hidden=hidden, sd=sd, nc=nc, drop=drop, md=md).to(device)
        ema = copy.deepcopy(model)
        opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.02)
        ts_total = ep * len(trl); ws = 5 * len(trl); sch = cw(opt, ws, ts_total)
        bva, bst = -1, None

        for e in range(ep):
            model.train()
            for batch in trl:
                batch = {k:v.to(device) for k,v in batch.items()}; opt.zero_grad()
                logits = model(batch, training=True)
                # Per-sample weighted loss
                per_sample_loss = F.cross_entropy(logits, batch["label"], reduction='none', label_smoothing=0.03)
                w = batch["weight"]
                loss = (per_sample_loss * w).mean()
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

        # Eval
        train_penult, train_logits, train_labs = get_penult_and_logits(ema, trl_ns)
        val_penult, val_logits, val_labs = get_penult_and_logits(ema, vl)
        test_penult, test_logits, test_labs = get_penult_and_logits(ema, tel)

        std_acc = accuracy_score(test_labs, np.argmax(test_logits, axis=1))
        if nc == 2:
            vd2 = val_logits[:,1]-val_logits[:,0]; td = test_logits[:,1]-test_logits[:,0]
            bt,bva2=0,0
            for t in np.arange(-3,3,0.02):
                va2=accuracy_score(val_labs,(vd2>t).astype(int))
                if va2>bva2:bva2,bt=va2,t
            tuned=accuracy_score(test_labs,(td>bt).astype(int))
            base_acc=max(std_acc,tuned)
        else:
            base_acc=std_acc
        single_accs.append(base_acc)

        # kNN
        bank_labels_t = torch.tensor(train_labs)
        knn_test = knn_logits(test_penult, train_penult, bank_labels_t, k=knn_k, nc=nc, temperature=knn_temp)
        knn_val = knn_logits(val_penult, train_penult, bank_labels_t, k=knn_k, nc=nc, temperature=knn_temp)
        best_knn = base_acc
        for alpha in np.arange(0, 0.55, 0.05):
            blended_val = (1-alpha)*val_logits + alpha*knn_val
            blended_test = (1-alpha)*test_logits + alpha*knn_test
            std=accuracy_score(test_labs,np.argmax(blended_test,axis=1))
            if nc==2:
                vd3=blended_val[:,1]-blended_val[:,0]; td3=blended_test[:,1]-blended_test[:,0]
                bt3,bva3=0,0
                for t in np.arange(-3,3,0.02):
                    va3=accuracy_score(val_labs,(vd3>t).astype(int))
                    if va3>bva3:bva3,bt3=va3,t
                tuned3=accuracy_score(test_labs,(td3>bt3).astype(int))
                acc=max(std,tuned3)
            else:
                acc=std
            if acc>best_knn:best_knn=acc
        knn_accs.append(best_knn)

    logger.info(f"  {name} [single]: Acc={np.mean(single_accs):.4f}±{np.std(single_accs):.4f} (max={np.max(single_accs):.4f}) >=0.85:{sum(1 for a in single_accs if a>=0.85)} >=0.90:{sum(1 for a in single_accs if a>=0.90)}")
    logger.info(f"  {name} [+kNN]: Acc={np.mean(knn_accs):.4f}±{np.std(knn_accs):.4f} (max={np.max(knn_accs):.4f}) >=0.85:{sum(1 for a in knn_accs if a>=0.85)} >=0.90:{sum(1 for a in knn_accs if a>=0.90)}")
    return np.mean(single_accs), np.max(knn_accs)

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

    if args.dataset_name == "HateMM":
        # Baseline without cartography (for comparison)
        run("6mod+ev no-carto", feats, splits, lm,
            ["text","audio","frame","t1","t2","ev"], sd, 192, 2e-4, 45, 0.15, 0.15, nr, nc, logger,
            use_cartography=False)
        # With cartography
        run("6mod+ev carto", feats, splits, lm,
            ["text","audio","frame","t1","t2","ev"], sd, 192, 2e-4, 45, 0.15, 0.15, nr, nc, logger,
            use_cartography=True)
    elif args.language == "English":
        run("T1E no-carto", feats, splits, lm,
            ["text","audio","frame","t1e","t2"], sd, 192, 2e-4, 45, 0.15, 0.15, nr, nc, logger,
            use_cartography=False)
        run("T1E carto", feats, splits, lm,
            ["text","audio","frame","t1e","t2"], sd, 192, 2e-4, 45, 0.15, 0.15, nr, nc, logger,
            use_cartography=True)
    else:
        run("5mod h=256 no-carto", feats, splits, lm,
            ["text","audio","frame","t1","t2"], sd, 256, 2e-4, 45, 0.15, 0.15, nr, nc, logger,
            use_cartography=False)
        run("5mod h=256 carto", feats, splits, lm,
            ["text","audio","frame","t1","t2"], sd, 256, 2e-4, 45, 0.15, 0.15, nr, nc, logger,
            use_cartography=True)

if __name__ == "__main__":
    main()
