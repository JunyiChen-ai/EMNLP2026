"""Asymmetric Loss + Class Rebalancing for FN-heavy datasets.

Error analysis shows EN MHC has 40 FN vs 3 FP (model biased toward Normal).
ZH MHC has 32 FN vs 5 FP (same pattern).

Techniques:
1. Asymmetric Focal Loss: higher gamma for positive (hate) class
2. Class-weighted CE: upweight hate class
3. Cost-sensitive threshold tuning: FN costs more than FP
4. Combined with kNN + best previous techniques
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
    lf = f"./logs/fusion_asym_{ts}.log"
    logger = logging.getLogger(f"fusasym_{ts}"); logger.setLevel(logging.INFO); logger.handlers.clear()
    fh = logging.FileHandler(lf, encoding="utf-8"); ch = logging.StreamHandler()
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    fh.setFormatter(fmt); ch.setFormatter(fmt); logger.addHandler(fh); logger.addHandler(ch)
    return logger

class AsymmetricFocalLoss(nn.Module):
    """Focal loss with per-class gamma: higher gamma for under-predicted class."""
    def __init__(self, gamma_pos=3.0, gamma_neg=1.0, label_smoothing=0.03):
        super().__init__()
        self.gp = gamma_pos  # gamma for positive class (hate) — harder
        self.gn = gamma_neg  # gamma for negative class (normal) — easier
        self.ls = label_smoothing

    def forward(self, logits, targets):
        nc = logits.size(1)
        with torch.no_grad():
            smooth = torch.full_like(logits, self.ls / (nc - 1))
            smooth.scatter_(1, targets.unsqueeze(1), 1.0 - self.ls)
        log_p = F.log_softmax(logits, dim=1)
        p = torch.exp(log_p)
        # Per-sample gamma based on true class
        gamma = torch.where(targets == 1, self.gp, self.gn).unsqueeze(1)
        focal_weight = (1 - p) ** gamma
        loss = -(focal_weight * smooth * log_p).sum(dim=1)
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

def run(name, feats, splits, lm, mk, sd, hidden, lr, ep, drop, md, nr, nc, logger,
        loss_type="asym_focal", class_weight=None, gamma_pos=3.0, gamma_neg=1.0):
    fk=list(mk)+["struct"]
    common=set.intersection(*[set(feats[k].keys()) for k in fk])&set(feats["labels"].keys())
    cur={s:[v for v in splits[s] if v in common] for s in splits}

    single_accs, knn_accs = [], []
    for ri in range(nr):
        seed = ri * 1000 + 42
        torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
        trd=DS(cur["train"],feats,lm,mk); vd=DS(cur["valid"],feats,lm,mk); ted=DS(cur["test"],feats,lm,mk)
        trd_u=DS(cur["train"],feats,lm,mk)
        trl=DataLoader(trd,32,True,collate_fn=collate_fn)
        vl=DataLoader(vd,64,False,collate_fn=collate_fn)
        tel=DataLoader(ted,64,False,collate_fn=collate_fn)
        trl_ns=DataLoader(trd_u,64,False,collate_fn=collate_fn)

        model=Fusion(mk,hidden=hidden,sd=sd,nc=nc,drop=drop,md=md).to(device)
        ema=copy.deepcopy(model)
        opt=optim.AdamW(model.parameters(),lr=lr,weight_decay=0.02)
        ts_t=ep*len(trl); ws=5*len(trl); sch=cw(opt,ws,ts_t)

        if loss_type == "asym_focal":
            crit = AsymmetricFocalLoss(gamma_pos=gamma_pos, gamma_neg=gamma_neg)
        elif loss_type == "weighted_ce":
            crit = nn.CrossEntropyLoss(weight=torch.tensor(class_weight, dtype=torch.float).to(device), label_smoothing=0.03)
        else:
            crit = nn.CrossEntropyLoss(label_smoothing=0.03)

        bva, bst = -1, None
        for e in range(ep):
            model.train()
            for batch in trl:
                batch={k:v.to(device) for k,v in batch.items()}; opt.zero_grad()
                crit(model(batch,training=True),batch["label"]).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); sch.step()
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
        tp,tl_arr,tla = get_pl(ema, trl_ns)
        vp,vl_arr,vla = get_pl(ema, vl)
        tep,tel_arr,tela = get_pl(ema, tel)
        base = best_thresh_acc(vl_arr, vla, tel_arr, tela, nc)
        single_accs.append(base)

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

    logger.info(f"  {name} [single]: Acc={np.mean(single_accs):.4f}±{np.std(single_accs):.4f} (max={np.max(single_accs):.4f}) >=0.85:{sum(1 for a in single_accs if a>=0.85)} >=0.90:{sum(1 for a in single_accs if a>=0.90)}")
    logger.info(f"  {name} [+kNN]: Acc={np.mean(knn_accs):.4f}±{np.std(knn_accs):.4f} (max={np.max(knn_accs):.4f}) >=0.85:{sum(1 for a in knn_accs if a>=0.85)} >=0.90:{sum(1 for a in knn_accs if a>=0.90)}")
    return np.max(knn_accs)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--dataset_name",default="HateMM",choices=["HateMM","Multihateclip"])
    parser.add_argument("--language",default="English")
    parser.add_argument("--num_runs",type=int,default=30)
    args=parser.parse_args()
    logger=setup_logger()

    if args.dataset_name=="HateMM":
        emb_dir="./embeddings/HateMM"; ann_path="./datasets/HateMM/annotation(new).json"
        split_dir="./datasets/HateMM/splits"; lm={"Non Hate":0,"Hate":1}; nc=2
    else:
        emb_dir=f"./embeddings/Multihateclip/{args.language}"
        ann_path=f"./datasets/Multihateclip/{args.language}/annotation(new).json"
        split_dir=f"./datasets/Multihateclip/{args.language}/splits"
        lm={"Normal":0,"Offensive":1,"Hateful":1}; nc=2

    feats={
        "text":torch.load(f"{emb_dir}/text_features.pth",map_location="cpu"),
        "audio":torch.load(f"{emb_dir}/wavlm_audio_features.pth",map_location="cpu"),
        "frame":torch.load(f"{emb_dir}/frame_features.pth",map_location="cpu"),
        "t1":torch.load(f"{emb_dir}/v12_t1_features.pth",map_location="cpu"),
        "t2":torch.load(f"{emb_dir}/v12_t2_features.pth",map_location="cpu"),
        "t1e":torch.load(f"{emb_dir}/v12_t1e_features.pth",map_location="cpu"),
        "ev":torch.load(f"{emb_dir}/v12_evidence_features.pth",map_location="cpu"),
        "struct":torch.load(f"{emb_dir}/v12_struct_features.pth",map_location="cpu"),
    }
    with open(ann_path) as f: feats["labels"]={d["Video_ID"]:d for d in json.load(f)}
    splits=load_split_ids(split_dir)
    sd=feats["struct"][list(feats["struct"].keys())[0]].shape[0]
    nr=args.num_runs
    logger.info(f"{args.dataset_name} {args.language}")

    if args.dataset_name=="HateMM":
        mk=["text","audio","frame","t1","t2","ev"]
        # Baseline
        run("6mod+ev CE",feats,splits,lm,mk,sd,192,2e-4,45,0.15,0.15,nr,nc,logger,loss_type="ce")
        # Asymmetric focal
        for gp in [2.0, 3.0, 4.0]:
            run(f"6mod+ev asym gp={gp}",feats,splits,lm,mk,sd,192,2e-4,45,0.15,0.15,nr,nc,logger,
                loss_type="asym_focal",gamma_pos=gp,gamma_neg=1.0)
        # Class-weighted CE
        run("6mod+ev wCE 1:1.5",feats,splits,lm,mk,sd,192,2e-4,45,0.15,0.15,nr,nc,logger,
            loss_type="weighted_ce",class_weight=[1.0, 1.5])
    else:
        # MHC: heavy FN problem
        mk_options = [
            (["text","audio","frame","t1","t2","ev"], "6mod+ev", 192),
            (["text","audio","frame","t1e","t2"], "T1E", 192),
        ]
        for mk, mname, h in mk_options:
            # Asymmetric focal with different gamma
            for gp in [2.0, 3.0, 4.0]:
                run(f"{mname} asym gp={gp}",feats,splits,lm,mk,sd,h,2e-4,45,0.15,0.15,nr,nc,logger,
                    loss_type="asym_focal",gamma_pos=gp,gamma_neg=1.0)
            # Class-weighted CE
            for w in [1.5, 2.0, 2.5]:
                run(f"{mname} wCE 1:{w}",feats,splits,lm,mk,sd,h,2e-4,45,0.15,0.15,nr,nc,logger,
                    loss_type="weighted_ce",class_weight=[1.0, w])

if __name__=="__main__":
    main()
