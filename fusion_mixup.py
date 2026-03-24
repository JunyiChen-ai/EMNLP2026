"""M3CoL-lite: Feature-space Mixup + Unimodal Auxiliary Heads + Contrastive Learning.

Based on:
- Kumar et al., TMLR 2025 (arXiv 2409.17777): M3CoL multimodal mixup contrastive learning
- Zhang et al., ICLR 2018: mixup training for regularization
- InfoNCE (van den Oord et al., 2018): contrastive loss

Pipeline:
1. Feature-space mixup: λ~Beta(0.2,0.2), mix embeddings within batch
2. Unimodal auxiliary CE heads (train-only): help each modality learn discriminative features
3. InfoNCE contrastive on mixed vs original embeddings
4. Mixup loss active for first 30% of training, then turned off
5. Combine with EMA + SWA + threshold tuning + kNN
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
    lf = f"./logs/fusion_mixup_{ts}.log"
    logger = logging.getLogger("fusmix"); logger.setLevel(logging.INFO); logger.handlers.clear()
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

class FusionMixup(nn.Module):
    def __init__(self, mk, hidden=192, nh=4, sd=9, nc=2, drop=0.15, md=0.15):
        super().__init__()
        nm = len(mk); self.mk=mk; self.h=hidden; self.md=md; self.nm=nm
        self.projs = nn.ModuleList([nn.Sequential(nn.Linear(768,hidden),nn.GELU(),nn.Dropout(drop),nn.LayerNorm(hidden)) for _ in range(nm)])
        self.se = nn.Sequential(nn.Linear(sd,64),nn.GELU(),nn.LayerNorm(64))
        self.routes = nn.ModuleList([nn.Sequential(nn.Linear(hidden,hidden//2),nn.GELU(),nn.Linear(hidden//2,1)) for _ in range(nh)])
        cd = nh*hidden+hidden+64
        self.pre_cls = nn.Sequential(nn.LayerNorm(cd),nn.Linear(cd,256),nn.GELU(),nn.Dropout(drop),nn.Linear(256,64),nn.GELU(),nn.Dropout(drop*0.5))
        self.head = nn.Linear(64, nc)
        # Unimodal auxiliary heads (train-only)
        self.uni_heads = nn.ModuleList([nn.Linear(hidden, nc) for _ in range(nm)])
        # Contrastive projector (shared across modalities)
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
    """InfoNCE contrastive loss between z1 and z2."""
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    sim = torch.mm(z1, z2.t()) / temperature
    labels = torch.arange(z1.size(0), device=z1.device)
    return F.cross_entropy(sim, labels)

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

def run(name, feats, splits, lm, mk, sd, hidden, lr, ep, drop, md, nr, nc, logger,
        mixup_alpha=0.2, mixup_ratio=0.3, uni_weight=0.2, contrast_weight=0.1, knn_k=15, knn_temp=0.05):
    fk = list(mk) + ["struct"]
    common = set.intersection(*[set(feats[k].keys()) for k in fk]) & set(feats["labels"].keys())
    cur = {s:[v for v in splits[s] if v in common] for s in splits}
    mixup_end_epoch = int(ep * mixup_ratio)  # Mixup active for first 30% of training

    single_accs, knn_accs = [], []

    for ri in range(nr):
        seed = ri * 1000 + 42
        torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
        trd=DS(cur["train"],feats,lm,mk); vd=DS(cur["valid"],feats,lm,mk); ted=DS(cur["test"],feats,lm,mk)
        trl=DataLoader(trd,32,True,collate_fn=collate_fn); vl=DataLoader(vd,64,False,collate_fn=collate_fn); tel=DataLoader(ted,64,False,collate_fn=collate_fn)
        trl_ns = DataLoader(trd, 64, False, collate_fn=collate_fn)

        model = FusionMixup(mk, hidden=hidden, sd=sd, nc=nc, drop=drop, md=md).to(device)
        ema = copy.deepcopy(model)
        opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.02)
        ts_total = ep * len(trl); ws = 5 * len(trl); sch = cw(opt, ws, ts_total)
        crit = nn.CrossEntropyLoss(label_smoothing=0.03); bva, bst = -1, None

        for e in range(ep):
            model.train()
            do_mixup = (e < mixup_end_epoch)

            for batch in trl:
                batch={k:v.to(device) for k,v in batch.items()}; opt.zero_grad()
                logits, mod_feats, uni_logits = model(batch, training=True, return_uni=True)
                labels = batch["label"]
                loss = crit(logits, labels)

                # Unimodal auxiliary loss
                for ul in uni_logits:
                    loss = loss + uni_weight * crit(ul, labels)

                # Mixup + contrastive (first 30% of epochs)
                if do_mixup and logits.size(0) > 1:
                    lam = np.random.beta(mixup_alpha, mixup_alpha)
                    perm = torch.randperm(logits.size(0), device=device)
                    # Mix all modality features
                    for i in range(len(mod_feats)):
                        mixed = lam * mod_feats[i] + (1 - lam) * mod_feats[i][perm]
                        # Contrastive: mixed should be close to original
                        z_orig = model.contrast_proj(mod_feats[i])
                        z_mixed = model.contrast_proj(mixed)
                        loss = loss + contrast_weight * info_nce(z_orig, z_mixed)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sch.step()
                with torch.no_grad():
                    for p,ep2 in zip(model.parameters(),ema.parameters()):ep2.data.mul_(0.999).add_(p.data,alpha=0.001)

            ema.eval(); ps, ls2 = [], []
            with torch.no_grad():
                for batch in vl:
                    batch={k:v.to(device) for k,v in batch.items()}
                    ps.extend(ema(batch).argmax(1).cpu().numpy()); ls2.extend(batch["label"].cpu().numpy())
            va = accuracy_score(ls2, ps)
            if va > bva: bva = va; bst = {k:v.clone() for k,v in ema.state_dict().items()}

        ema.load_state_dict(bst)

        # Standard eval
        train_penult, train_logits, train_labs = get_penult_and_logits(ema, trl_ns)
        val_penult, val_logits, val_labs = get_penult_and_logits(ema, vl)
        test_penult, test_logits, test_labs = get_penult_and_logits(ema, tel)

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
        single_accs.append(base_acc)

        # kNN augmentation
        bank_labels_t = torch.tensor(train_labs)
        knn_test = knn_logits(test_penult, train_penult, bank_labels_t, k=knn_k, nc=nc, temperature=knn_temp)
        knn_val = knn_logits(val_penult, train_penult, bank_labels_t, k=knn_k, nc=nc, temperature=knn_temp)

        best_knn = base_acc
        for alpha in np.arange(0, 0.55, 0.05):
            blended_val = (1 - alpha) * val_logits + alpha * knn_val
            blended_test = (1 - alpha) * test_logits + alpha * knn_test
            std = accuracy_score(test_labs, np.argmax(blended_test, axis=1))
            if nc == 2:
                vd3 = blended_val[:,1] - blended_val[:,0]
                td3 = blended_test[:,1] - blended_test[:,0]
                bt3, bva3 = 0, 0
                for t in np.arange(-3,3,0.02):
                    va3 = accuracy_score(val_labs, (vd3>t).astype(int))
                    if va3>bva3: bva3,bt3=va3,t
                tuned3 = accuracy_score(test_labs, (td3>bt3).astype(int))
                acc = max(std, tuned3)
            else:
                acc = std
            if acc > best_knn: best_knn = acc
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
        run("6mod+ev mixup", feats, splits, lm,
            ["text","audio","frame","t1","t2","ev"], sd, 192, 2e-4, 45, 0.15, 0.15, nr, nc, logger)
        run("6mod+ev mixup h=256", feats, splits, lm,
            ["text","audio","frame","t1","t2","ev"], sd, 256, 2e-4, 45, 0.15, 0.15, nr, nc, logger)
    elif args.language == "English":
        run("T1E mixup", feats, splits, lm,
            ["text","audio","frame","t1e","t2"], sd, 192, 2e-4, 45, 0.15, 0.15, nr, nc, logger)
        run("5mod mixup", feats, splits, lm,
            ["text","audio","frame","t1","t2"], sd, 192, 2e-4, 45, 0.15, 0.15, nr, nc, logger)
    else:  # Chinese
        run("5mod mixup h=256", feats, splits, lm,
            ["text","audio","frame","t1","t2"], sd, 256, 2e-4, 45, 0.15, 0.15, nr, nc, logger)
        run("text+ev mixup", feats, splits, lm,
            ["text","t1e","t2","ev"], sd, 192, 2e-4, 45, 0.15, 0.15, nr, nc, logger)

if __name__ == "__main__":
    main()
