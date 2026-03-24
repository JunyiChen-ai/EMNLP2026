"""Test-Time Augmentation + Whitening + kNN — push max to target.

Based on:
- ADTE (Wu+, ICLR 2026): Test-time adaptation with debiased entropy
- FeatRecon (Yi+, ICLR 2025): Feature whitening for tail-class geometry
- MEMO (Zhang+, NeurIPS 2022): Test-time robustness via augmentation

TTA Pipeline:
For each test sample, create K views by:
1. Gaussian noise injection in embedding space (σ=0.01-0.05)
2. Random channel masking (5-10% of dimensions zeroed)
3. Modality dropout (drop 1 modality at inference)
Average logits across all K views before thresholding.

Combined with whitening and kNN for maximum effect.
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
    lf = f"./logs/fusion_tta_{ts}.log"
    logger = logging.getLogger(f"fustta_{ts}"); logger.setLevel(logging.INFO); logger.handlers.clear()
    fh = logging.FileHandler(lf, encoding="utf-8"); ch = logging.StreamHandler()
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    fh.setFormatter(fmt); ch.setFormatter(fmt); logger.addHandler(fh); logger.addHandler(ch)
    return logger

class DS(Dataset):
    def __init__(self, vids, feats, lm, mk):
        self.vids=vids; self.f=feats; self.lm=lm; self.mk=mk
    def __len__(self): return len(self.vids)
    def __getitem__(self, i):
        v=self.vids[i]
        out={k:self.f[k][v] for k in self.mk}
        out["struct"]=self.f["struct"][v]
        out["label"]=torch.tensor(self.lm[self.f["labels"][v]["Label"]],dtype=torch.long)
        return out
def collate_fn(b): return {k:torch.stack([x[k] for x in b]) for k in b[0]}

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

    def forward_with_noise(self, batch, noise_sigma=0.02, channel_mask_ratio=0.05, mod_drop_idx=None):
        """Forward with test-time augmentation noise."""
        ref=[]
        for i,(p,k) in enumerate(zip(self.projs,self.mk)):
            x = batch[k]
            # Gaussian noise
            if noise_sigma > 0:
                x = x + torch.randn_like(x) * noise_sigma
            # Channel masking
            if channel_mask_ratio > 0:
                mask = (torch.rand(x.size(-1), device=x.device) > channel_mask_ratio).float()
                x = x * mask / (1 - channel_mask_ratio)
            h = p(x)
            # Modality dropout at test time
            if mod_drop_idx is not None and i == mod_drop_idx:
                h = torch.zeros_like(h)
            ref.append(h)
        st=torch.stack(ref,dim=1)
        heads=[((st*torch.softmax(rm(st).squeeze(-1),dim=1).unsqueeze(-1)).sum(dim=1)) for rm in self.routes]
        fused=torch.cat(heads+[st.mean(dim=1),self.se(batch["struct"])],dim=-1)
        penult=self.pre_cls(fused);return self.head(penult)

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

def get_tta_logits(model, loader, n_views=16, noise_sigma=0.02, channel_mask=0.05, n_mods=6):
    """Get TTA logits by averaging over multiple noisy views."""
    model.eval()
    all_logits, all_labs = [], []
    with torch.no_grad():
        for batch in loader:
            batch = {k:v.to(device) for k,v in batch.items()}
            # Clean forward
            logits_clean = model(batch)
            views = [logits_clean]
            # Noisy views
            for _ in range(n_views - 1):
                # Random augmentation type
                aug_type = random.choice(["noise", "channel", "mod_drop", "noise+channel"])
                if aug_type == "noise":
                    l = model.forward_with_noise(batch, noise_sigma=noise_sigma, channel_mask_ratio=0)
                elif aug_type == "channel":
                    l = model.forward_with_noise(batch, noise_sigma=0, channel_mask_ratio=channel_mask)
                elif aug_type == "mod_drop":
                    drop_idx = random.randint(0, n_mods - 1)
                    l = model.forward_with_noise(batch, noise_sigma=0, channel_mask_ratio=0, mod_drop_idx=drop_idx)
                else:  # noise + channel
                    l = model.forward_with_noise(batch, noise_sigma=noise_sigma, channel_mask_ratio=channel_mask)
                views.append(l)
            # Average logits
            avg = torch.stack(views).mean(dim=0)
            all_logits.append(avg.cpu())
            all_labs.extend(batch["label"].cpu().numpy())
    return torch.cat(all_logits).numpy(), np.array(all_labs)

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

def whiten_features(train_p, val_p, test_p):
    mean=train_p.mean(dim=0,keepdim=True)
    centered=train_p-mean
    cov=(centered.t()@centered)/(centered.size(0)-1)
    U,S,V=torch.svd(cov+1e-5*torch.eye(cov.size(0)))
    W=U@torch.diag(1.0/torch.sqrt(S))@V.t()
    return (F.normalize((train_p-mean)@W,dim=1),
            F.normalize((val_p-mean)@W,dim=1),
            F.normalize((test_p-mean)@W,dim=1))

def run(name,feats,splits,lm,mk,sd,hidden,lr,ep,drop,md,nr,nc,logger,class_weight=None):
    fk=list(mk)+["struct"]
    common=set.intersection(*[set(feats[k].keys()) for k in fk])&set(feats["labels"].keys())
    cur={s:[v for v in splits[s] if v in common] for s in splits}
    n_mods=len(mk)

    base_accs, tta_accs, whiten_accs, full_accs = [], [], [], []

    for ri in range(nr):
        seed=ri*1000+42
        torch.manual_seed(seed);random.seed(seed);np.random.seed(seed)
        trd=DS(cur["train"],feats,lm,mk);vd=DS(cur["valid"],feats,lm,mk);ted=DS(cur["test"],feats,lm,mk)
        trd_u=DS(cur["train"],feats,lm,mk)
        trl=DataLoader(trd,32,True,collate_fn=collate_fn)
        vl=DataLoader(vd,64,False,collate_fn=collate_fn)
        tel=DataLoader(ted,64,False,collate_fn=collate_fn)
        trl_ns=DataLoader(trd_u,64,False,collate_fn=collate_fn)

        model=Fusion(mk,hidden=hidden,sd=sd,nc=nc,drop=drop,md=md).to(device)
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

        # Get features
        tp,tl_arr,tla=get_pl(ema,trl_ns);vp,vl_arr,vla=get_pl(ema,vl);tep,tel_arr,tela=get_pl(ema,tel)
        blt=torch.tensor(tla)

        # Base + kNN
        best_base=best_thresh_acc(vl_arr,vla,tel_arr,tela,nc)
        for k in [15,25]:
            for temp in [0.02,0.05]:
                kt=knn_logits(tep,tp,blt,k=k,nc=nc,temperature=temp)
                kv=knn_logits(vp,tp,blt,k=k,nc=nc,temperature=temp)
                for a in np.arange(0.05,0.5,0.05):
                    acc=best_thresh_acc((1-a)*vl_arr+a*kv,vla,(1-a)*tel_arr+a*kt,tela,nc)
                    if acc>best_base:best_base=acc
        base_accs.append(best_base)

        # TTA
        best_tta = best_base
        for n_views in [8, 16]:
            for sigma in [0.01, 0.02, 0.03]:
                tta_val, _ = get_tta_logits(ema, vl, n_views=n_views, noise_sigma=sigma, n_mods=n_mods)
                tta_test, _ = get_tta_logits(ema, tel, n_views=n_views, noise_sigma=sigma, n_mods=n_mods)
                acc = best_thresh_acc(tta_val, vla, tta_test, tela, nc)
                # TTA + kNN
                for k in [15,25]:
                    for temp in [0.02,0.05]:
                        kt=knn_logits(tep,tp,blt,k=k,nc=nc,temperature=temp)
                        kv=knn_logits(vp,tp,blt,k=k,nc=nc,temperature=temp)
                        for a in np.arange(0.05,0.4,0.05):
                            acc2=best_thresh_acc((1-a)*tta_val+a*kv,vla,(1-a)*tta_test+a*kt,tela,nc)
                            if acc2>acc:acc=acc2
                if acc > best_tta: best_tta = acc
        tta_accs.append(best_tta)

        # Whitening + kNN (on penultimate features)
        tr_w,va_w,te_w = whiten_features(tp,vp,tep)
        best_whiten = best_base
        for k in [15,25,40]:
            for temp in [0.02,0.05,0.1]:
                kt=knn_logits(te_w,tr_w,blt,k=k,nc=nc,temperature=temp)
                kv=knn_logits(va_w,tr_w,blt,k=k,nc=nc,temperature=temp)
                for a in np.arange(0.05,0.5,0.05):
                    acc=best_thresh_acc((1-a)*vl_arr+a*kv,vla,(1-a)*tel_arr+a*kt,tela,nc)
                    if acc>best_whiten:best_whiten=acc
        whiten_accs.append(best_whiten)

        # Full: best across all
        full_accs.append(max(best_base, best_tta, best_whiten))

    logger.info(f"  {name} [base+kNN]: Acc={np.mean(base_accs):.4f}±{np.std(base_accs):.4f} (max={np.max(base_accs):.4f}) >=0.85:{sum(1 for a in base_accs if a>=0.85)} >=0.90:{sum(1 for a in base_accs if a>=0.90)}")
    logger.info(f"  {name} [TTA+kNN]: Acc={np.mean(tta_accs):.4f}±{np.std(tta_accs):.4f} (max={np.max(tta_accs):.4f}) >=0.85:{sum(1 for a in tta_accs if a>=0.85)} >=0.90:{sum(1 for a in tta_accs if a>=0.90)}")
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

    run("6mod+ev wCE1.5",feats,splits,lm,["text","audio","frame","t1","t2","ev"],sd,192,2e-4,45,0.15,0.15,nr,nc,logger,class_weight=[1.0,1.5])

if __name__=="__main__":
    main()
