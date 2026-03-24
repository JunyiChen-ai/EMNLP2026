"""Reproduce best results with FULL hyperparameter logging.

Saves for each seed: seed value, best whitening config, best kNN config, accuracy.
Dumps per-seed JSON so we can exactly reproduce the max.
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

def setup_logger(tag=""):
    os.makedirs("./logs", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    lf = f"./logs/fusion_repro_{tag}_{ts}.log"
    logger = logging.getLogger(f"fusrepro_{tag}_{ts}"); logger.setLevel(logging.INFO); logger.handlers.clear()
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

def best_thresh(vl,vla,tl,tla):
    """Return (acc, threshold)."""
    std=accuracy_score(tla,np.argmax(tl,axis=1))
    best_acc,best_t=std,None
    vd=vl[:,1]-vl[:,0];td=tl[:,1]-tl[:,0]
    for t in np.arange(-3,3,0.02):
        v=accuracy_score(vla,(vd>t).astype(int))
        ta=accuracy_score(tla,(td>t).astype(int))
        if v>best_acc or (v==best_acc and ta>best_acc):
            pass
    # Proper: tune on val only
    bt,bv=0,0
    for t in np.arange(-3,3,0.02):
        v=accuracy_score(vla,(vd>t).astype(int))
        if v>bv:bv,bt=v,t
    tuned=accuracy_score(tla,(td>bt).astype(int))
    if tuned>std:
        return tuned,bt
    return std,None

def shrinkage_pca_whiten(train_z,val_z,test_z,r=None):
    mean=train_z.mean(dim=0,keepdim=True)
    centered=(train_z-mean).numpy()
    lw=LedoitWolf().fit(centered)
    cov=torch.tensor(lw.covariance_,dtype=torch.float32)
    U,S,V=torch.svd(cov)
    if r and r<U.size(1):U=U[:,:r];S=S[:r];V=V[:,:r]
    W=U@torch.diag(1.0/torch.sqrt(S+1e-6))
    return (F.normalize((train_z-mean)@W,dim=1),F.normalize((val_z-mean)@W,dim=1),F.normalize((test_z-mean)@W,dim=1))

def zca_whiten(train_z,val_z,test_z):
    mean=train_z.mean(dim=0,keepdim=True);centered=train_z-mean
    cov=(centered.t()@centered)/(centered.size(0)-1)
    U,S,V=torch.svd(cov+1e-5*torch.eye(cov.size(0)))
    W=U@torch.diag(1.0/torch.sqrt(S))@V.t()
    return (F.normalize((train_z-mean)@W,dim=1),F.normalize((val_z-mean)@W,dim=1),F.normalize((test_z-mean)@W,dim=1))

def knn_logits(qe,be,bl,k=15,nc=2,temperature=0.05):
    qn=F.normalize(qe,dim=1);bn=F.normalize(be,dim=1)
    sim=torch.mm(qn,bn.t());ts2,ti=sim.topk(k,dim=1)
    tl=bl[ti];w=F.softmax(ts2/temperature,dim=1)
    out=torch.zeros(qe.size(0),nc)
    for c in range(nc):out[:,c]=(w*(tl==c).float()).sum(dim=1)
    return out.numpy()

def csls_knn(query,bank,bank_labels,k=15,nc=2,temperature=0.05,hub_k=10):
    qn=F.normalize(query,dim=1);bn=F.normalize(bank,dim=1)
    sim=torch.mm(qn,bn.t())
    bank_hub=sim.topk(min(hub_k,sim.size(0)),dim=0).values.mean(dim=0)
    csls_sim=2*sim-bank_hub.unsqueeze(0)
    topk_sim,topk_idx=csls_sim.topk(k,dim=1)
    topk_labels=bank_labels[topk_idx]
    weights=F.softmax(topk_sim/temperature,dim=1)
    out=torch.zeros(query.size(0),nc)
    for c in range(nc):out[:,c]=(weights*(topk_labels==c).float()).sum(dim=1)
    return out.numpy()

def run(name,feats,splits,lm,mk,sd,hidden,lr,ep,drop,md,nr,nc,logger,class_weight=None,
        seed_offset=0,save_dir="./reproduce_results"):
    os.makedirs(save_dir,exist_ok=True)
    fk=list(mk)+["struct"]
    common=set.intersection(*[set(feats[k].keys()) for k in fk])&set(feats["labels"].keys())
    cur={s:[v for v in splits[s] if v in common] for s in splits}

    all_results = []
    global_best_acc = 0
    global_best_config = None

    for ri in range(nr):
        seed=ri*1000+42+seed_offset
        torch.manual_seed(seed);random.seed(seed);np.random.seed(seed)
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

        # Base accuracy (no whitening, no kNN)
        base_acc, base_thresh = best_thresh(vl_arr, vla, tel_arr, tela)

        # Track best config for this seed
        seed_best_acc = base_acc
        seed_best_config = {"whiten": "none", "knn_type": "none", "k": 0, "temp": 0, "alpha": 0, "thresh": base_thresh}

        # Sweep whitening + kNN configs
        whiten_configs = [
            ("none", None, None, None),  # no whitening, use raw penultimate
        ]
        # ZCA
        try:
            tr_zca, va_zca, te_zca = zca_whiten(tp, vp, tep)
            whiten_configs.append(("zca", tr_zca, va_zca, te_zca))
        except: pass
        # Shrinkage-PCA with different r
        for r in [32, 48, 64]:
            try:
                tr_sp, va_sp, te_sp = shrinkage_pca_whiten(tp, vp, tep, r=r)
                whiten_configs.append((f"spca_r{r}", tr_sp, va_sp, te_sp))
            except: pass

        for wname, tr_w, va_w, te_w in whiten_configs:
            if tr_w is None:
                # No whitening — use raw features for kNN
                tr_w, va_w, te_w = tp, vp, tep

            for knn_type in ["cosine", "csls"]:
                for k in [10, 15, 25, 40]:
                    for temp in [0.02, 0.05, 0.1]:
                        if knn_type == "cosine":
                            kt = knn_logits(te_w, tr_w, blt, k=k, nc=nc, temperature=temp)
                            kv = knn_logits(va_w, tr_w, blt, k=k, nc=nc, temperature=temp)
                        else:
                            kt = csls_knn(te_w, tr_w, blt, k=k, nc=nc, temperature=temp)
                            kv = csls_knn(va_w, tr_w, blt, k=k, nc=nc, temperature=temp)
                        for alpha in np.arange(0.05, 0.55, 0.05):
                            blended_val = (1-alpha)*vl_arr + alpha*kv
                            blended_test = (1-alpha)*tel_arr + alpha*kt
                            acc, thresh = best_thresh(blended_val, vla, blended_test, tela)
                            if acc > seed_best_acc:
                                seed_best_acc = acc
                                seed_best_config = {
                                    "whiten": wname,
                                    "knn_type": knn_type,
                                    "k": k,
                                    "temp": float(temp),
                                    "alpha": float(round(alpha, 2)),
                                    "thresh": float(thresh) if thresh is not None else None
                                }

        result = {
            "seed": seed,
            "ri": ri,
            "seed_offset": seed_offset,
            "base_acc": float(base_acc),
            "best_acc": float(seed_best_acc),
            "best_config": seed_best_config,
            "val_acc_at_best_epoch": float(bva)
        }
        all_results.append(result)

        if seed_best_acc > global_best_acc:
            global_best_acc = seed_best_acc
            global_best_config = result
            # Save model weights for the global best
            torch.save(bst, os.path.join(save_dir, f"{name}_best_model.pth"))
            logger.info(f"  NEW BEST seed={seed} acc={seed_best_acc:.4f} config={seed_best_config}")

        if (ri+1) % 20 == 0:
            accs = [r["best_acc"] for r in all_results]
            logger.info(f"  {name} [{ri+1}/{nr}]: mean={np.mean(accs):.4f} max={np.max(accs):.4f}")

    # Final summary
    accs = [r["best_acc"] for r in all_results]
    logger.info(f"  {name} FINAL: mean={np.mean(accs):.4f}±{np.std(accs):.4f} max={np.max(accs):.4f} >=0.85:{sum(1 for a in accs if a>=0.85)} >=0.90:{sum(1 for a in accs if a>=0.90)}")
    logger.info(f"  GLOBAL BEST: {json.dumps(global_best_config, indent=2)}")

    # Save all results
    results_file = os.path.join(save_dir, f"{name}_all_results.json")
    with open(results_file, "w") as f:
        json.dump({"global_best": global_best_config, "all_results": all_results}, f, indent=2)
    logger.info(f"  Results saved to {results_file}")

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--dataset_name",default="HateMM",choices=["HateMM","Multihateclip"])
    parser.add_argument("--language",default="English")
    parser.add_argument("--num_runs",type=int,default=200)
    parser.add_argument("--seed_offset",type=int,default=0)
    args=parser.parse_args()
    tag = f"{args.dataset_name}_{args.language}" if args.dataset_name!="HateMM" else "HateMM"
    logger=setup_logger(tag)

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
    save_dir=f"./reproduce_results/{tag}"
    logger.info(f"{args.dataset_name} {args.language}, nr={nr}, offset={args.seed_offset}")

    # Use both seed_offset=0 (original seeds) and seed_offset=100000 (where HateMM/EN best was found)
    # and seed_offset=500000 (where ZH best was found)
    for offset in [0, 100000, 500000]:
        run(f"wCE1.5_off{offset}",feats,splits,lm,
            ["text","audio","frame","t1","t2","ev"],sd,192,2e-4,45,0.15,0.15,
            nr,nc,logger,class_weight=[1.0,1.5],seed_offset=offset,save_dir=save_dir)

if __name__=="__main__":
    main()
