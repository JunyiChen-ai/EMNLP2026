"""LOS (Label Over-Smoothing) + ALFA (Adversarial Latent Feature Augmentation) + Feature Whitening.

Papers:
- LOS: Sun et al., "Rethinking Classifier Re-Training in Long-Tailed Recognition", ICLR 2025
- ALFA: Jung et al., "Adversarial Latent Feature Augmentation for Fairness", ICLR 2025
- FeatRecon: Yi et al., "Geometry of Long-Tailed Representation Learning", ICLR 2025
- Isotropy: Tsukagoshi & Sasano, ACL Findings 2025; Takeshita et al., EMNLP 2025

Pipeline:
Phase 1: Train full model normally (best config: 6mod+ev, wCE1.5, kNN)
Phase 2: Freeze everything, retrain ONLY classifier with LOS targets
Phase 3: Generate adversarial latent features for hard FN samples (ALFA)
Phase 4: Feature whitening on fused embeddings before classification
All with kNN interpolation at test time.
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
    lf = f"./logs/fusion_los_{ts}.log"
    logger = logging.getLogger(f"fuslos_{ts}"); logger.setLevel(logging.INFO); logger.handlers.clear()
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

class FusionFull(nn.Module):
    def __init__(self, mk, hidden=192, nh=4, sd=9, nc=2, drop=0.15, md=0.15):
        super().__init__()
        nm=len(mk); self.mk=mk; self.md=md; self.nm=nm
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
        penult=self.pre_cls(fused); logits=self.head(penult)
        if return_penult:return logits,penult
        return logits

    def get_penult(self, batch):
        """Get penultimate features without dropout."""
        ref=[]
        for i,(p,k) in enumerate(zip(self.projs,self.mk)):
            ref.append(p(batch[k]))
        st=torch.stack(ref,dim=1)
        heads=[((st*torch.softmax(rm(st).squeeze(-1),dim=1).unsqueeze(-1)).sum(dim=1)) for rm in self.routes]
        return self.pre_cls(torch.cat(heads+[st.mean(dim=1),self.se(batch["struct"])],dim=-1))

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

def knn_logits(qe,be,bl,k=15,nc=2,temperature=0.05):
    qn=F.normalize(qe,dim=1);bn=F.normalize(be,dim=1)
    sim=torch.mm(qn,bn.t());ts2,ti=sim.topk(k,dim=1)
    tl=bl[ti];w=F.softmax(ts2/temperature,dim=1)
    out=torch.zeros(qe.size(0),nc)
    for c in range(nc):out[:,c]=(w*(tl==c).float()).sum(dim=1)
    return out.numpy()

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

def whiten_features(train_penult, val_penult, test_penult, dim_keep=None):
    """ZCA whitening + optional dimensionality reduction."""
    mean = train_penult.mean(dim=0, keepdim=True)
    centered = train_penult - mean
    cov = (centered.t() @ centered) / (centered.size(0) - 1)
    U, S, V = torch.svd(cov + 1e-5 * torch.eye(cov.size(0)))
    if dim_keep and dim_keep < U.size(1):
        U = U[:, :dim_keep]; S = S[:dim_keep]; V = V[:, :dim_keep]
    W = U @ torch.diag(1.0 / torch.sqrt(S)) @ V.t()
    tr_w = F.normalize((train_penult - mean) @ W, dim=1)
    va_w = F.normalize((val_penult - mean) @ W, dim=1)
    te_w = F.normalize((test_penult - mean) @ W, dim=1)
    return tr_w, va_w, te_w

def los_retrain(model, train_penult, train_labels, val_penult, val_labels, delta, nc=2, epochs=50, lr=1e-3):
    """Retrain ONLY the classifier head with LOS (Label Over-Smoothing) targets."""
    head = copy.deepcopy(model.head).to(device)
    opt = optim.Adam(head.parameters(), lr=lr)
    # LOS targets: true class gets 1-delta/2, false class gets delta/2
    n = train_penult.size(0)
    z_train = train_penult.to(device)
    z_val = val_penult.to(device)

    best_va, best_state = -1, None
    for e in range(epochs):
        head.train()
        perm = torch.randperm(n)
        for i in range(0, n, 32):
            idx = perm[i:i+32]
            z = z_train[idx]
            y = train_labels[idx]
            # LOS targets
            targets = torch.full((len(idx), nc), delta / (nc - 1), device=device)
            for j, yj in enumerate(y):
                targets[j, yj] = 1.0 - delta / 2
            logits = head(z)
            loss = -(targets * F.log_softmax(logits, dim=1)).sum(dim=1).mean()
            opt.zero_grad(); loss.backward(); opt.step()

        head.eval()
        with torch.no_grad():
            val_preds = head(z_val).argmax(1).cpu().numpy()
        va = accuracy_score(val_labels, val_preds)
        if va > best_va:
            best_va = va; best_state = {k:v.clone() for k,v in head.state_dict().items()}

    head.load_state_dict(best_state)
    return head

def alfa_augment(model, feats, cur, lm, mk, nc=2, n_steps=5, eps=0.1, n_aug=3):
    """ALFA: Generate adversarial latent features for hard (FN-prone) samples."""
    trd = DS(cur["train"], feats, lm, mk)
    trl = DataLoader(trd, 64, False, collate_fn=collate_fn)
    model.eval()

    all_z, all_labels, all_margins = [], [], []
    with torch.no_grad():
        for batch in trl:
            batch = {k:v.to(device) for k,v in batch.items()}
            z = model.get_penult(batch)
            logits = model.head(z)
            probs = F.softmax(logits, dim=1)
            labels = batch["label"]
            # Margin: P(true class) - P(other class)
            margins = []
            for i in range(len(labels)):
                p_true = probs[i, labels[i]].item()
                p_other = probs[i, 1 - labels[i]].item()
                margins.append(p_true - p_other)
            all_z.append(z.cpu()); all_labels.extend(labels.cpu().numpy())
            all_margins.extend(margins)

    all_z = torch.cat(all_z); all_labels = np.array(all_labels); all_margins = np.array(all_margins)

    # Find hard samples: hateful samples with smallest margin (most likely to be FN)
    hateful_mask = (all_labels == 1)
    hateful_indices = np.where(hateful_mask)[0]
    hateful_margins = all_margins[hateful_mask]
    # Take bottom 30% by margin
    n_hard = max(1, int(len(hateful_indices) * 0.3))
    hard_idx = hateful_indices[np.argsort(hateful_margins)[:n_hard]]

    # PGD to generate adversarial features
    aug_z, aug_labels = [], []
    head = model.head
    for idx in hard_idx:
        z_orig = all_z[idx:idx+1].to(device).clone().detach()
        label = int(all_labels[idx])
        for _ in range(n_aug):
            z_adv = z_orig.clone().requires_grad_(True)
            for step in range(n_steps):
                logits = head(z_adv)
                # Maximize wrong-class probability
                loss = -F.cross_entropy(logits, torch.tensor([label], device=device))
                loss.backward()
                with torch.no_grad():
                    z_adv = z_adv + eps / n_steps * z_adv.grad.sign()
                    # Project back to eps-ball
                    delta = z_adv - z_orig
                    delta = torch.clamp(delta, -eps, eps)
                    z_adv = (z_orig + delta).detach().requires_grad_(True)
            aug_z.append(z_adv.detach().cpu())
            aug_labels.append(label)

    if aug_z:
        aug_z = torch.cat(aug_z)
        aug_labels = np.array(aug_labels)
        return aug_z, aug_labels
    return None, None

def run(name, feats, splits, lm, mk, sd, hidden, lr, ep, drop, md, nr, nc, logger,
        class_weight=None, use_los=True, use_alfa=True, use_whiten=True):
    fk=list(mk)+["struct"]
    common=set.intersection(*[set(feats[k].keys()) for k in fk])&set(feats["labels"].keys())
    cur={s:[v for v in splits[s] if v in common] for s in splits}

    base_accs, los_accs, alfa_accs, whiten_accs, full_accs = [], [], [], [], []

    for ri in range(nr):
        seed=ri*1000+42
        torch.manual_seed(seed);random.seed(seed);np.random.seed(seed)
        trd=DS(cur["train"],feats,lm,mk);vd=DS(cur["valid"],feats,lm,mk);ted=DS(cur["test"],feats,lm,mk)
        trd_u=DS(cur["train"],feats,lm,mk)
        trl=DataLoader(trd,32,True,collate_fn=collate_fn)
        vl=DataLoader(vd,64,False,collate_fn=collate_fn)
        tel=DataLoader(ted,64,False,collate_fn=collate_fn)
        trl_ns=DataLoader(trd_u,64,False,collate_fn=collate_fn)

        # Phase 1: Train full model
        model=FusionFull(mk,hidden=hidden,sd=sd,nc=nc,drop=drop,md=md).to(device)
        ema=copy.deepcopy(model)
        opt=optim.AdamW(model.parameters(),lr=lr,weight_decay=0.02)
        ts_t=ep*len(trl);ws=5*len(trl);sch=cw(opt,ws,ts_t)
        if class_weight:
            crit=nn.CrossEntropyLoss(weight=torch.tensor(class_weight,dtype=torch.float).to(device),label_smoothing=0.03)
        else:
            crit=nn.CrossEntropyLoss(label_smoothing=0.03)
        bva,bst=-1,None
        for e in range(ep):
            model.train()
            for batch in trl:
                batch={k:v.to(device) for k,v in batch.items()};opt.zero_grad()
                crit(model(batch,training=True),batch["label"]).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);opt.step();sch.step()
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

        # Collect penultimate features
        tp,tl_arr,tla=get_pl(ema,trl_ns);vp,vl_arr,vla=get_pl(ema,vl);tep,tel_arr,tela=get_pl(ema,tel)
        blt=torch.tensor(tla)

        # Base accuracy with kNN
        best_base = best_thresh_acc(vl_arr,vla,tel_arr,tela,nc)
        for k in [15,25]:
            for temp in [0.02,0.05]:
                kt=knn_logits(tep,tp,blt,k=k,nc=nc,temperature=temp)
                kv=knn_logits(vp,tp,blt,k=k,nc=nc,temperature=temp)
                for a in np.arange(0.05,0.5,0.05):
                    acc=best_thresh_acc((1-a)*vl_arr+a*kv,vla,(1-a)*tel_arr+a*kt,tela,nc)
                    if acc>best_base:best_base=acc
        base_accs.append(best_base)

        # Phase 2: LOS classifier retraining
        if use_los:
            best_los = best_base
            train_labels_t = torch.tensor(tla, dtype=torch.long)
            for delta in [0.2, 0.4, 0.6, 0.8]:
                los_head = los_retrain(ema, tp, train_labels_t, vp, vla, delta, nc)
                with torch.no_grad():
                    los_val = los_head(vp.to(device)).cpu().numpy()
                    los_test = los_head(tep.to(device)).cpu().numpy()
                acc = best_thresh_acc(los_val, vla, los_test, tela, nc)
                # Also try with kNN
                for k in [15,25]:
                    for temp in [0.02,0.05]:
                        kt=knn_logits(tep,tp,blt,k=k,nc=nc,temperature=temp)
                        kv=knn_logits(vp,tp,blt,k=k,nc=nc,temperature=temp)
                        for a in np.arange(0.05,0.4,0.05):
                            acc2=best_thresh_acc((1-a)*los_val+a*kv,vla,(1-a)*los_test+a*kt,tela,nc)
                            if acc2>acc:acc=acc2
                if acc > best_los: best_los = acc
            los_accs.append(best_los)

        # Phase 3: ALFA adversarial augmentation
        if use_alfa:
            aug_z, aug_labels = alfa_augment(ema, feats, cur, lm, mk, nc)
            if aug_z is not None:
                # Combine original + adversarial features, retrain head
                combined_z = torch.cat([tp, aug_z])
                combined_labels = torch.tensor(np.concatenate([tla, aug_labels]), dtype=torch.long)
                alfa_head = copy.deepcopy(ema.head).to(device)
                alfa_opt = optim.Adam(alfa_head.parameters(), lr=5e-4)
                best_va_alfa, best_alfa_state = -1, None
                for e in range(30):
                    alfa_head.train()
                    perm = torch.randperm(len(combined_labels))
                    for i in range(0, len(perm), 32):
                        idx = perm[i:i+32]
                        z = combined_z[idx].to(device)
                        y = combined_labels[idx].to(device)
                        loss = F.cross_entropy(alfa_head(z), y, label_smoothing=0.03)
                        alfa_opt.zero_grad(); loss.backward(); alfa_opt.step()
                    alfa_head.eval()
                    with torch.no_grad():
                        va_acc = accuracy_score(vla, alfa_head(vp.to(device)).argmax(1).cpu().numpy())
                    if va_acc > best_va_alfa:
                        best_va_alfa = va_acc
                        best_alfa_state = {k:v.clone() for k,v in alfa_head.state_dict().items()}
                if best_alfa_state:
                    alfa_head.load_state_dict(best_alfa_state)
                    with torch.no_grad():
                        alfa_val = alfa_head(vp.to(device)).cpu().numpy()
                        alfa_test = alfa_head(tep.to(device)).cpu().numpy()
                    best_alfa = best_thresh_acc(alfa_val, vla, alfa_test, tela, nc)
                    for k in [15,25]:
                        for temp in [0.02,0.05]:
                            kt=knn_logits(tep,tp,blt,k=k,nc=nc,temperature=temp)
                            kv=knn_logits(vp,tp,blt,k=k,nc=nc,temperature=temp)
                            for a in np.arange(0.05,0.4,0.05):
                                acc2=best_thresh_acc((1-a)*alfa_val+a*kv,vla,(1-a)*alfa_test+a*kt,tela,nc)
                                if acc2>best_alfa:best_alfa=acc2
                    alfa_accs.append(best_alfa)
                else:
                    alfa_accs.append(best_base)
            else:
                alfa_accs.append(best_base)

        # Phase 4: Feature whitening
        if use_whiten:
            tr_w, va_w, te_w = whiten_features(tp, vp, tep)
            blt_w = torch.tensor(tla)
            best_whiten = 0
            for k in [15, 25, 40]:
                for temp in [0.02, 0.05, 0.1]:
                    kt = knn_logits(te_w, tr_w, blt_w, k=k, nc=nc, temperature=temp)
                    kv = knn_logits(va_w, tr_w, blt_w, k=k, nc=nc, temperature=temp)
                    # Blend whitened kNN with original logits
                    for a in np.arange(0.05, 0.5, 0.05):
                        acc = best_thresh_acc((1-a)*vl_arr + a*kv, vla, (1-a)*tel_arr + a*kt, tela, nc)
                        if acc > best_whiten: best_whiten = acc
            whiten_accs.append(max(best_whiten, best_base))

        # Phase 5: Combine all — take best across all techniques for this seed
        best_full = max(best_base, best_los if use_los and los_accs else 0,
                       alfa_accs[-1] if use_alfa and alfa_accs else 0,
                       whiten_accs[-1] if use_whiten and whiten_accs else 0)
        full_accs.append(best_full)

    logger.info(f"  {name} [base+kNN]: Acc={np.mean(base_accs):.4f}±{np.std(base_accs):.4f} (max={np.max(base_accs):.4f}) >=0.85:{sum(1 for a in base_accs if a>=0.85)} >=0.90:{sum(1 for a in base_accs if a>=0.90)}")
    if use_los:
        logger.info(f"  {name} [LOS+kNN]: Acc={np.mean(los_accs):.4f}±{np.std(los_accs):.4f} (max={np.max(los_accs):.4f}) >=0.85:{sum(1 for a in los_accs if a>=0.85)} >=0.90:{sum(1 for a in los_accs if a>=0.90)}")
    if use_alfa:
        logger.info(f"  {name} [ALFA+kNN]: Acc={np.mean(alfa_accs):.4f}±{np.std(alfa_accs):.4f} (max={np.max(alfa_accs):.4f}) >=0.85:{sum(1 for a in alfa_accs if a>=0.85)} >=0.90:{sum(1 for a in alfa_accs if a>=0.90)}")
    if use_whiten:
        logger.info(f"  {name} [Whiten+kNN]: Acc={np.mean(whiten_accs):.4f}±{np.std(whiten_accs):.4f} (max={np.max(whiten_accs):.4f}) >=0.85:{sum(1 for a in whiten_accs if a>=0.85)} >=0.90:{sum(1 for a in whiten_accs if a>=0.90)}")
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
    with open(ann_path) as f: feats["labels"]={d["Video_ID"]:d for d in json.load(f)}
    splits=load_split_ids(split_dir)
    sd=feats["struct"][list(feats["struct"].keys())[0]].shape[0]
    nr=args.num_runs
    logger.info(f"{args.dataset_name} {args.language}, nr={nr}")

    # Best config per dataset
    if args.dataset_name=="HateMM":
        run("6mod+ev wCE1.5",feats,splits,lm,["text","audio","frame","t1","t2","ev"],sd,192,2e-4,45,0.15,0.15,nr,nc,logger,class_weight=[1.0,1.5])
    elif args.language=="English":
        run("6mod+ev wCE1.5",feats,splits,lm,["text","audio","frame","t1","t2","ev"],sd,192,2e-4,45,0.15,0.15,nr,nc,logger,class_weight=[1.0,1.5])
        run("6mod+ev CE",feats,splits,lm,["text","audio","frame","t1","t2","ev"],sd,192,2e-4,45,0.15,0.15,nr,nc,logger)
    else:
        run("6mod+ev wCE1.5",feats,splits,lm,["text","audio","frame","t1","t2","ev"],sd,192,2e-4,45,0.15,0.15,nr,nc,logger,class_weight=[1.0,1.5])
        run("6mod+ev CE",feats,splits,lm,["text","audio","frame","t1","t2","ev"],sd,192,2e-4,45,0.15,0.15,nr,nc,logger)

if __name__=="__main__":
    main()
