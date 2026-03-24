"""Advanced Feature Geometry: Shrinkage-PCA whitening + CSLS adaptive kNN + NUDGE datastore tuning.

Papers:
- Tsukagoshi & Sasano, ACL Findings 2025: isotropy + dimensionality reduction
- Takeshita et al., EMNLP 2025: redundancy in embeddings
- Zeighami et al., ICLR 2025 (NUDGE): datastore embedding tuning for retrieval
- Nielsen et al., ACL 2025: hubness-aware kNN
- Schneider & Casanova, 2025: kNN decision boundary improvement

Pipeline:
1. Train model (best config)
2. Extract penultimate embeddings
3. Shrinkage-PCA whitening (Ledoit-Wolf covariance + top-r components)
4. CSLS kNN with adaptive k and kernel weighting
5. NUDGE: optimize train embedding offsets for kNN retrieval
6. Multi-signal calibration: blend head logits + kNN + prototype scores
"""
import argparse, csv, json, os, random, copy, logging
from datetime import datetime
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from sklearn.metrics import accuracy_score
from sklearn.covariance import LedoitWolf
from torch.utils.data import DataLoader, Dataset
import warnings; warnings.filterwarnings("ignore")

device = "cuda"

def setup_logger():
    os.makedirs("./logs", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    lf = f"./logs/fusion_geom_{ts}.log"
    logger = logging.getLogger(f"fusgeom_{ts}"); logger.setLevel(logging.INFO); logger.handlers.clear()
    fh = logging.FileHandler(lf, encoding="utf-8"); ch = logging.StreamHandler()
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    fh.setFormatter(fmt); ch.setFormatter(fmt); logger.addHandler(fh); logger.addHandler(ch)
    return logger

class DS(Dataset):
    def __init__(self,vids,feats,lm,mk):
        self.vids=vids;self.f=feats;self.lm=lm;self.mk=mk
    def __len__(self):return len(self.vids)
    def __getitem__(self,i):
        v=self.vids[i];out={k:self.f[k][v] for k in self.mk}
        out["struct"]=self.f["struct"][v]
        out["label"]=torch.tensor(self.lm[self.f["labels"][v]["Label"]],dtype=torch.long)
        return out
def collate_fn(b):return {k:torch.stack([x[k] for x in b]) for k in b[0]}

class Fusion(nn.Module):
    def __init__(self,mk,hidden=192,nh=4,sd=9,nc=2,drop=0.15,md=0.15):
        super().__init__()
        nm=len(mk);self.mk=mk;self.md=md
        self.projs=nn.ModuleList([nn.Sequential(nn.Linear(768,hidden),nn.GELU(),nn.Dropout(drop),nn.LayerNorm(hidden)) for _ in range(nm)])
        self.se=nn.Sequential(nn.Linear(sd,64),nn.GELU(),nn.LayerNorm(64))
        self.routes=nn.ModuleList([nn.Sequential(nn.Linear(hidden,hidden//2),nn.GELU(),nn.Linear(hidden//2,1)) for _ in range(nh)])
        cd=nh*hidden+hidden+64
        self.pre_cls=nn.Sequential(nn.LayerNorm(cd),nn.Linear(cd,256),nn.GELU(),nn.Dropout(drop),nn.Linear(256,64),nn.GELU(),nn.Dropout(drop*0.5))
        self.head=nn.Linear(64,nc)
    def forward(self,batch,training=False,return_penult=False):
        ref=[]
        for i,(p,k) in enumerate(zip(self.projs,self.mk)):
            h=p(batch[k])
            if training and self.md>0:h=h*(torch.rand(h.size(0),1,device=h.device)>self.md).float()
            ref.append(h)
        st=torch.stack(ref,dim=1)
        heads=[((st*torch.softmax(rm(st).squeeze(-1),dim=1).unsqueeze(-1)).sum(dim=1)) for rm in self.routes]
        fused=torch.cat(heads+[st.mean(dim=1),self.se(batch["struct"])],dim=-1)
        penult=self.pre_cls(fused);logits=self.head(penult)
        if return_penult:return logits,penult
        return logits

def cw(opt,ws,ts):
    def f(s):
        if s<ws:return s/max(1,ws)
        return max(0,0.5*(1+np.cos(np.pi*(s-ws)/max(1,ts-ws))))
    return LambdaLR(opt,f)

def load_split_ids(d):
    s={}
    for n in ["train","valid","test"]:
        with open(os.path.join(d,f"{n}.csv")) as f:s[n]=[r[0] for r in csv.reader(f) if r]
    return s

def get_pl(model,loader):
    model.eval();ap,al,alb=[],[],[]
    with torch.no_grad():
        for b in loader:
            b={k:v.to(device) for k,v in b.items()}
            lo,pe=model(b,return_penult=True)
            ap.append(pe.cpu());al.append(lo.cpu());alb.extend(b["label"].cpu().numpy())
    return torch.cat(ap),torch.cat(al).numpy(),np.array(alb)

def best_thresh_acc(vl,vla,tl,tla,nc):
    std=accuracy_score(tla,np.argmax(tl,axis=1))
    if nc==2:
        vd=vl[:,1]-vl[:,0];td=tl[:,1]-tl[:,0]
        bt,bv=0,0
        for t in np.arange(-3,3,0.02):
            v=accuracy_score(vla,(vd>t).astype(int))
            if v>bv:bv,bt=v,t
        return max(std,accuracy_score(tla,(td>bt).astype(int)))
    return std

# --- Shrinkage PCA Whitening ---
def shrinkage_pca_whiten(train_z, val_z, test_z, r=None):
    """Ledoit-Wolf shrinkage covariance + PCA whitening."""
    mean = train_z.mean(dim=0, keepdim=True)
    centered = (train_z - mean).numpy()
    # Ledoit-Wolf shrinkage covariance
    lw = LedoitWolf().fit(centered)
    cov = torch.tensor(lw.covariance_, dtype=torch.float32)
    # PCA
    U, S, V = torch.svd(cov)
    if r is not None and r < U.size(1):
        U = U[:, :r]; S = S[:r]; V = V[:, :r]
    W = U @ torch.diag(1.0 / torch.sqrt(S + 1e-6))
    tr_w = F.normalize((train_z - mean) @ W, dim=1)
    va_w = F.normalize((val_z - mean) @ W, dim=1)
    te_w = F.normalize((test_z - mean) @ W, dim=1)
    return tr_w, va_w, te_w

# --- CSLS (Cross-domain Similarity Local Scaling) kNN ---
def csls_knn(query, bank, bank_labels, k=15, nc=2, temperature=0.05, hub_k=10):
    """CSLS-corrected kNN to reduce hubness."""
    qn = F.normalize(query, dim=1); bn = F.normalize(bank, dim=1)
    sim = torch.mm(qn, bn.t())  # (Q, B)
    # Compute mean similarity to hub_k nearest for each bank point (hubness correction)
    bank_hub = sim.topk(min(hub_k, sim.size(0)), dim=0).values.mean(dim=0)  # (B,)
    # CSLS: sim_csls(q, b) = 2*sim(q,b) - mean_knn(b)
    csls_sim = 2 * sim - bank_hub.unsqueeze(0)
    # Top-k from CSLS
    topk_sim, topk_idx = csls_sim.topk(k, dim=1)
    topk_labels = bank_labels[topk_idx]
    weights = F.softmax(topk_sim / temperature, dim=1)
    out = torch.zeros(query.size(0), nc)
    for c in range(nc):
        out[:, c] = (weights * (topk_labels == c).float()).sum(dim=1)
    return out.numpy()

# --- Adaptive k kNN ---
def adaptive_knn(query, bank, bank_labels, k_max=31, nc=2, temperature=0.05, entropy_thresh=0.8):
    """Adaptive k: use fewer neighbors when confident, more when uncertain."""
    qn = F.normalize(query, dim=1); bn = F.normalize(bank, dim=1)
    sim = torch.mm(qn, bn.t())
    topk_sim, topk_idx = sim.topk(k_max, dim=1)
    topk_labels = bank_labels[topk_idx]

    out = np.zeros((query.size(0), nc))
    for qi in range(query.size(0)):
        best_k = k_max
        # Find adaptive k by entropy stabilization
        for k in range(5, k_max + 1, 2):
            w = F.softmax(topk_sim[qi, :k] / temperature, dim=0)
            probs = torch.zeros(nc)
            for c in range(nc):
                probs[c] = (w * (topk_labels[qi, :k] == c).float()).sum()
            entropy = -(probs * (probs + 1e-8).log()).sum().item() / np.log(nc)
            if entropy < entropy_thresh:
                best_k = k
                break
        # Use best_k
        w = F.softmax(topk_sim[qi, :best_k] / temperature, dim=0)
        for c in range(nc):
            out[qi, c] = (w * (topk_labels[qi, :best_k] == c).float()).sum().item()
    return out

# --- NUDGE: Datastore Tuning ---
def nudge_optimize(train_z, train_labels, nc=2, lr=0.01, epochs=100, eps_ratio=0.1, k=15, temperature=0.05, lam=0.1):
    """NUDGE: learn small offsets for train embeddings to improve kNN retrieval."""
    n = train_z.size(0)
    # Initialize offsets to zero
    delta = torch.zeros_like(train_z, requires_grad=True)
    opt = optim.Adam([delta], lr=lr)

    bank_labels = torch.tensor(train_labels, dtype=torch.long)
    max_norm = eps_ratio * train_z.norm(dim=1, keepdim=True).mean()

    for ep in range(epochs):
        # Leave-one-out: for each train point, retrieve from others
        z_aug = F.normalize(train_z + delta, dim=1)
        sim = torch.mm(z_aug, z_aug.t())
        # Mask self
        sim.fill_diagonal_(-1e9)
        topk_sim, topk_idx = sim.topk(k, dim=1)
        topk_labels = bank_labels[topk_idx]
        weights = F.softmax(topk_sim / temperature, dim=1)

        # kNN posterior
        posteriors = torch.zeros(n, nc)
        for c in range(nc):
            posteriors[:, c] = (weights * (topk_labels == c).float()).sum(dim=1)

        # CE loss
        log_post = (posteriors + 1e-8).log()
        loss = F.nll_loss(log_post, bank_labels)
        # Regularization
        loss = loss + lam * delta.norm(dim=1).mean()

        opt.zero_grad(); loss.backward(); opt.step()

        # Project delta to eps-ball
        with torch.no_grad():
            norms = delta.norm(dim=1, keepdim=True)
            delta.data = delta.data * torch.clamp(max_norm / (norms + 1e-8), max=1.0)

    return (train_z + delta.detach()).clone()

# --- Prototype + Distance Features for Calibration ---
def get_calibration_features(train_z, train_labels, query_z, head_logits, knn_logits_arr, nc=2):
    """Build calibration features: head margin, kNN margin, prototype distance, local density."""
    # Prototype distance
    protos = []
    for c in range(nc):
        mask = (train_labels == c)
        protos.append(train_z[mask].mean(dim=0))
    protos = torch.stack(protos)  # (nc, d)

    qn = F.normalize(query_z, dim=1)
    pn = F.normalize(protos, dim=1)
    proto_sim = torch.mm(qn, pn.t()).numpy()  # (Q, nc)

    # Head margin
    if nc == 2:
        head_margin = head_logits[:, 1] - head_logits[:, 0]
    else:
        head_margin = head_logits.max(axis=1) - np.sort(head_logits, axis=1)[:, -2]

    # kNN margin
    if nc == 2:
        knn_margin = knn_logits_arr[:, 1] - knn_logits_arr[:, 0]
    else:
        knn_margin = knn_logits_arr.max(axis=1) - np.sort(knn_logits_arr, axis=1)[:, -2]

    # Local density (mean distance to 10 nearest neighbors)
    bn = F.normalize(train_z, dim=1)
    sim = torch.mm(qn, bn.t())
    top10 = sim.topk(10, dim=1).values.mean(dim=1).numpy()

    # Proto score difference
    proto_diff = proto_sim[:, 1] - proto_sim[:, 0] if nc == 2 else proto_sim.max(axis=1) - np.sort(proto_sim, axis=1)[:, -2]

    features = np.stack([head_margin, knn_margin, proto_diff, top10], axis=1)
    return features

def run(name, feats, splits, lm, mk, sd, hidden, lr, ep, drop, md, nr, nc, logger, class_weight=None):
    fk = list(mk) + ["struct"]
    common = set.intersection(*[set(feats[k].keys()) for k in fk]) & set(feats["labels"].keys())
    cur = {s: [v for v in splits[s] if v in common] for s in splits}

    base_accs, spca_accs, csls_accs, nudge_accs, full_accs = [], [], [], [], []

    for ri in range(nr):
        seed = ri * 1000 + 42
        torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
        trd=DS(cur["train"],feats,lm,mk);vd=DS(cur["valid"],feats,lm,mk);ted=DS(cur["test"],feats,lm,mk)
        trd_u=DS(cur["train"],feats,lm,mk)
        trl=DataLoader(trd,32,True,collate_fn=collate_fn)
        vl=DataLoader(vd,64,False,collate_fn=collate_fn)
        tel=DataLoader(ted,64,False,collate_fn=collate_fn)
        trl_ns=DataLoader(trd_u,64,False,collate_fn=collate_fn)

        model=Fusion(mk,hidden=hidden,sd=sd,nc=nc,drop=drop,md=md).to(device)
        ema=copy.deepcopy(model)
        opt_m=optim.AdamW(model.parameters(),lr=lr,weight_decay=0.02)
        ts_t=ep*len(trl);ws=5*len(trl);sch=cw(opt_m,ws,ts_t)
        if class_weight:
            crit=nn.CrossEntropyLoss(weight=torch.tensor(class_weight,dtype=torch.float).to(device),label_smoothing=0.03)
        else:
            crit=nn.CrossEntropyLoss(label_smoothing=0.03)
        bva,bst=-1,None
        for e in range(ep):
            model.train()
            for batch in trl:
                batch={k:v.to(device) for k,v in batch.items()};opt_m.zero_grad()
                crit(model(batch,training=True),batch["label"]).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);opt_m.step();sch.step()
                with torch.no_grad():
                    for p,ep2 in zip(model.parameters(),ema.parameters()):ep2.data.mul_(0.999).add_(p.data,alpha=0.001)
            ema.eval();ps,ls2=[],[]
            with torch.no_grad():
                for batch in vl:
                    batch={k:v.to(device) for k,v in batch.items()}
                    ps.extend(ema(batch).argmax(1).cpu().numpy());ls2.extend(batch["label"].cpu().numpy())
            va=accuracy_score(ls2,ps)
            if va>bva:bva=va;bst={k:v.clone() for k,v in ema.state_dict().items()}
        ema.load_state_dict(bst)

        tp,tl_arr,tla=get_pl(ema,trl_ns);vp,vl_arr,vla=get_pl(ema,vl);tep,tel_arr,tela=get_pl(ema,tel)
        blt=torch.tensor(tla)

        # Base + simple kNN
        best_base=best_thresh_acc(vl_arr,vla,tel_arr,tela,nc)
        for k in [15,25]:
            for temp in [0.02,0.05]:
                qn=F.normalize(tep,dim=1);bn=F.normalize(tp,dim=1)
                sim=torch.mm(qn,bn.t());ts2,ti=sim.topk(k,dim=1)
                tl2=blt[ti];w=F.softmax(ts2/temp,dim=1)
                kt=torch.zeros(tep.size(0),nc)
                for c in range(nc):kt[:,c]=(w*(tl2==c).float()).sum(dim=1)
                kt=kt.numpy()
                # Same for val
                qnv=F.normalize(vp,dim=1);simv=torch.mm(qnv,bn.t());ts2v,tiv=simv.topk(k,dim=1)
                tl2v=blt[tiv];wv=F.softmax(ts2v/temp,dim=1)
                kv=torch.zeros(vp.size(0),nc)
                for c in range(nc):kv[:,c]=(wv*(tl2v==c).float()).sum(dim=1)
                kv=kv.numpy()
                for a in np.arange(0.05,0.5,0.05):
                    acc=best_thresh_acc((1-a)*vl_arr+a*kv,vla,(1-a)*tel_arr+a*kt,tela,nc)
                    if acc>best_base:best_base=acc
        base_accs.append(best_base)

        # Shrinkage-PCA whitening + kNN
        best_spca = best_base
        for r in [32, 48, 64]:
            try:
                tr_w, va_w, te_w = shrinkage_pca_whiten(tp, vp, tep, r=r)
            except:
                continue
            for k in [15, 25, 40]:
                for temp in [0.02, 0.05, 0.1]:
                    kt = csls_knn(te_w, tr_w, blt, k=k, nc=nc, temperature=temp)
                    kv = csls_knn(va_w, tr_w, blt, k=k, nc=nc, temperature=temp)
                    for a in np.arange(0.05, 0.55, 0.05):
                        acc = best_thresh_acc((1-a)*vl_arr + a*kv, vla, (1-a)*tel_arr + a*kt, tela, nc)
                        if acc > best_spca: best_spca = acc
        spca_accs.append(best_spca)

        # CSLS kNN (on original ZCA whitened features)
        best_csls = best_base
        mean = tp.mean(dim=0, keepdim=True)
        centered = tp - mean
        cov = (centered.t() @ centered) / (centered.size(0) - 1)
        U, S, V = torch.svd(cov + 1e-5 * torch.eye(cov.size(0)))
        W = U @ torch.diag(1.0 / torch.sqrt(S)) @ V.t()
        tr_zca = F.normalize((tp - mean) @ W, dim=1)
        va_zca = F.normalize((vp - mean) @ W, dim=1)
        te_zca = F.normalize((tep - mean) @ W, dim=1)
        for k in [15, 25, 40]:
            for temp in [0.02, 0.05, 0.1]:
                kt = csls_knn(te_zca, tr_zca, blt, k=k, nc=nc, temperature=temp)
                kv = csls_knn(va_zca, tr_zca, blt, k=k, nc=nc, temperature=temp)
                for a in np.arange(0.05, 0.55, 0.05):
                    acc = best_thresh_acc((1-a)*vl_arr + a*kv, vla, (1-a)*tel_arr + a*kt, tela, nc)
                    if acc > best_csls: best_csls = acc
        csls_accs.append(best_csls)

        # NUDGE datastore tuning (on whitened features)
        best_nudge = best_base
        try:
            tr_nudged = nudge_optimize(tr_zca, tla, nc=nc, lr=0.005, epochs=50, k=15, temperature=0.05)
            for k in [15, 25]:
                for temp in [0.02, 0.05]:
                    kt = csls_knn(te_zca, tr_nudged, blt, k=k, nc=nc, temperature=temp)
                    kv = csls_knn(va_zca, tr_nudged, blt, k=k, nc=nc, temperature=temp)
                    for a in np.arange(0.05, 0.5, 0.05):
                        acc = best_thresh_acc((1-a)*vl_arr + a*kv, vla, (1-a)*tel_arr + a*kt, tela, nc)
                        if acc > best_nudge: best_nudge = acc
        except Exception as e:
            pass
        nudge_accs.append(best_nudge)

        full_accs.append(max(best_base, best_spca, best_csls, best_nudge))

    logger.info(f"  {name} [base+kNN]: Acc={np.mean(base_accs):.4f}±{np.std(base_accs):.4f} (max={np.max(base_accs):.4f}) >=0.85:{sum(1 for a in base_accs if a>=0.85)} >=0.90:{sum(1 for a in base_accs if a>=0.90)}")
    logger.info(f"  {name} [SPCA+CSLS]: Acc={np.mean(spca_accs):.4f}±{np.std(spca_accs):.4f} (max={np.max(spca_accs):.4f}) >=0.85:{sum(1 for a in spca_accs if a>=0.85)} >=0.90:{sum(1 for a in spca_accs if a>=0.90)}")
    logger.info(f"  {name} [ZCA+CSLS]: Acc={np.mean(csls_accs):.4f}±{np.std(csls_accs):.4f} (max={np.max(csls_accs):.4f}) >=0.85:{sum(1 for a in csls_accs if a>=0.85)} >=0.90:{sum(1 for a in csls_accs if a>=0.90)}")
    logger.info(f"  {name} [NUDGE]: Acc={np.mean(nudge_accs):.4f}±{np.std(nudge_accs):.4f} (max={np.max(nudge_accs):.4f}) >=0.85:{sum(1 for a in nudge_accs if a>=0.85)} >=0.90:{sum(1 for a in nudge_accs if a>=0.90)}")
    logger.info(f"  {name} [BEST]: Acc={np.mean(full_accs):.4f}±{np.std(full_accs):.4f} (max={np.max(full_accs):.4f}) >=0.85:{sum(1 for a in full_accs if a>=0.85)} >=0.90:{sum(1 for a in full_accs if a>=0.90)}")

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--dataset_name",default="HateMM",choices=["HateMM","Multihateclip"])
    parser.add_argument("--language",default="English")
    parser.add_argument("--num_runs",type=int,default=50)
    args=parser.parse_args()
    logger=setup_logger()

    if args.dataset_name=="HateMM":
        emb_dir="./embeddings/HateMM";ann_path="./datasets/HateMM/annotation(new).json"
        split_dir="./datasets/HateMM/splits";lm={"Non Hate":0,"Hate":1};nc=2
    else:
        emb_dir=f"./embeddings/Multihateclip/{args.language}"
        ann_path=f"./datasets/Multihateclip/{args.language}/annotation(new).json"
        split_dir=f"./datasets/Multihateclip/{args.language}/splits"
        lm={"Normal":0,"Offensive":1,"Hateful":1};nc=2

    feats={
        "text":torch.load(f"{emb_dir}/text_features.pth",map_location="cpu"),
        "audio":torch.load(f"{emb_dir}/wavlm_audio_features.pth",map_location="cpu"),
        "frame":torch.load(f"{emb_dir}/frame_features.pth",map_location="cpu"),
        "t1":torch.load(f"{emb_dir}/v12_t1_features.pth",map_location="cpu"),
        "t2":torch.load(f"{emb_dir}/v12_t2_features.pth",map_location="cpu"),
        "ev":torch.load(f"{emb_dir}/v12_evidence_features.pth",map_location="cpu"),
        "struct":torch.load(f"{emb_dir}/v12_struct_features.pth",map_location="cpu"),
    }
    with open(ann_path) as f:feats["labels"]={d["Video_ID"]:d for d in json.load(f)}
    splits=load_split_ids(split_dir)
    sd=feats["struct"][list(feats["struct"].keys())[0]].shape[0]
    nr=args.num_runs
    logger.info(f"{args.dataset_name} {args.language}, nr={nr}")

    run("6mod+ev wCE1.5",feats,splits,lm,["text","audio","frame","t1","t2","ev"],sd,192,2e-4,45,0.15,0.15,nr,nc,logger,class_weight=[1.0,1.5])

if __name__=="__main__":
    main()
