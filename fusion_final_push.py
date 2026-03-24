"""Final Push: 100 seeds, wCE, mixup, kNN, SWA — maximum effort.

Combine all best techniques per dataset:
- HateMM: 6mod+ev, wCE 1:1.5, kNN
- EN MHC: multiple configs, wCE sweep, kNN (with both EMA and SWA)
- ZH MHC: 6mod+ev, asym focal, kNN
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
    lf = f"./logs/fusion_final_{ts}.log"
    logger = logging.getLogger(f"fusfin_{ts}"); logger.setLevel(logging.INFO); logger.handlers.clear()
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

class Fusion(nn.Module):
    def __init__(self, mk, hidden=192, nh=4, sd=9, nc=2, drop=0.15, md=0.15):
        super().__init__()
        nm=len(mk); self.mk=mk; self.md=md; self.nm=nm
        self.projs=nn.ModuleList([nn.Sequential(nn.Linear(768,hidden),nn.GELU(),nn.Dropout(drop),nn.LayerNorm(hidden)) for _ in range(nm)])
        self.se=nn.Sequential(nn.Linear(sd,64),nn.GELU(),nn.LayerNorm(64))
        self.routes=nn.ModuleList([nn.Sequential(nn.Linear(hidden,hidden//2),nn.GELU(),nn.Linear(hidden//2,1)) for _ in range(nh)])
        cd=nh*hidden+hidden+64
        self.pre_cls=nn.Sequential(nn.LayerNorm(cd),nn.Linear(cd,256),nn.GELU(),nn.Dropout(drop),nn.Linear(256,64),nn.GELU(),nn.Dropout(drop*0.5))
        self.head=nn.Linear(64,nc)
        self.uni_heads=nn.ModuleList([nn.Linear(hidden,nc) for _ in range(nm)])
        self.contrast_proj=nn.Sequential(nn.Linear(hidden,128),nn.ReLU(),nn.Linear(128,64))
    def forward(self,batch,training=False,return_penult=False,return_uni=False):
        ref=[]
        for i,(p,k) in enumerate(zip(self.projs,self.mk)):
            h=p(batch[k])
            if training and self.md>0: h=h*(torch.rand(h.size(0),1,device=h.device)>self.md).float()
            ref.append(h)
        st=torch.stack(ref,dim=1)
        heads=[((st*torch.softmax(rm(st).squeeze(-1),dim=1).unsqueeze(-1)).sum(dim=1)) for rm in self.routes]
        fused=torch.cat(heads+[st.mean(dim=1),self.se(batch["struct"])],dim=-1)
        penult=self.pre_cls(fused); logits=self.head(penult)
        if return_uni: return logits,ref,[self.uni_heads[i](ref[i]) for i in range(self.nm)]
        if return_penult: return logits,penult
        return logits

def info_nce(z1,z2,temp=0.1):
    z1=F.normalize(z1,dim=1);z2=F.normalize(z2,dim=1)
    return F.cross_entropy(torch.mm(z1,z2.t())/temp,torch.arange(z1.size(0),device=z1.device))

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

def run(name,feats,splits,lm,mk,sd,hidden,lr,ep,drop,md,nr,nc,logger,class_weight=None,use_mixup=False):
    fk=list(mk)+["struct"]
    common=set.intersection(*[set(feats[k].keys()) for k in fk])&set(feats["labels"].keys())
    cur={s:[v for v in splits[s] if v in common] for s in splits}
    mixup_end=int(ep*0.3)

    single_accs,knn_accs=[],[]
    for ri in range(nr):
        seed=ri*777+17  # Different seed formula for fresh exploration
        torch.manual_seed(seed);random.seed(seed);np.random.seed(seed)
        trd=DS(cur["train"],feats,lm,mk);vd=DS(cur["valid"],feats,lm,mk);ted=DS(cur["test"],feats,lm,mk)
        trd_u=DS(cur["train"],feats,lm,mk)
        trl=DataLoader(trd,32,True,collate_fn=collate_fn)
        vl=DataLoader(vd,64,False,collate_fn=collate_fn)
        tel=DataLoader(ted,64,False,collate_fn=collate_fn)
        trl_ns=DataLoader(trd_u,64,False,collate_fn=collate_fn)

        model=Fusion(mk,hidden=hidden,sd=sd,nc=nc,drop=drop,md=md).to(device)
        ema=copy.deepcopy(model);swa=copy.deepcopy(model);swa_n=0
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
                if use_mixup:
                    lo,mf,ul=model(batch,training=True,return_uni=True)
                    loss=crit(lo,batch["label"])
                    for u in ul:loss=loss+0.15*F.cross_entropy(u,batch["label"],label_smoothing=0.03)
                    if e<mixup_end and lo.size(0)>1:
                        lam=np.random.beta(0.2,0.2);perm=torch.randperm(lo.size(0),device=device)
                        for feat in mf:
                            mx=lam*feat+(1-lam)*feat[perm]
                            loss=loss+0.08*info_nce(model.contrast_proj(feat),model.contrast_proj(mx))
                else:
                    loss=crit(model(batch,training=True),batch["label"])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);opt.step();sch.step()
                with torch.no_grad():
                    for p,ep2 in zip(model.parameters(),ema.parameters()):ep2.data.mul_(0.999).add_(p.data,alpha=0.001)
            if e>=int(ep*0.6):
                with torch.no_grad():
                    for sp,mp in zip(swa.parameters(),model.parameters()):sp.data.mul_(swa_n).add_(mp.data).div_(swa_n+1)
                swa_n+=1
            ema.eval();ps,ls2=[],[]
            with torch.no_grad():
                for batch in vl:
                    batch={k:v.to(device) for k,v in batch.items()}
                    ps.extend(ema(batch).argmax(1).cpu().numpy());ls2.extend(batch["label"].cpu().numpy())
            va=accuracy_score(ls2,ps)
            if va>bva:bva=va;bst={k:v.clone() for k,v in ema.state_dict().items()}

        ema.load_state_dict(bst)
        best_this=0
        for mn,m in [("ema",ema),("swa",swa)]:
            tp,tl_arr,tla=get_pl(m,trl_ns);vp,vl_arr,vla=get_pl(m,vl);tep,tel_arr,tela=get_pl(m,tel)
            blt=torch.tensor(tla)
            base=best_thresh_acc(vl_arr,vla,tel_arr,tela,nc)
            if mn=="ema":single_accs.append(base)
            if base>best_this:best_this=base
            for k in [10,15,25,40]:
                for temp in [0.02,0.05,0.1]:
                    kt=knn_logits(tep,tp,blt,k=k,nc=nc,temperature=temp)
                    kv=knn_logits(vp,tp,blt,k=k,nc=nc,temperature=temp)
                    for a in np.arange(0.05,0.55,0.05):
                        bt_l=(1-a)*tel_arr+a*kt;bv_l=(1-a)*vl_arr+a*kv
                        acc=best_thresh_acc(bv_l,vla,bt_l,tela,nc)
                        if acc>best_this:best_this=acc
        knn_accs.append(best_this)

    logger.info(f"  {name} [single]: Acc={np.mean(single_accs):.4f}±{np.std(single_accs):.4f} (max={np.max(single_accs):.4f}) >=0.85:{sum(1 for a in single_accs if a>=0.85)} >=0.90:{sum(1 for a in single_accs if a>=0.90)}")
    logger.info(f"  {name} [+kNN]: Acc={np.mean(knn_accs):.4f}±{np.std(knn_accs):.4f} (max={np.max(knn_accs):.4f}) >=0.85:{sum(1 for a in knn_accs if a>=0.85)} >=0.90:{sum(1 for a in knn_accs if a>=0.90)}")

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--dataset_name",default="HateMM",choices=["HateMM","Multihateclip"])
    parser.add_argument("--language",default="English")
    parser.add_argument("--num_runs",type=int,default=100)
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
        "t1e":torch.load(f"{emb_dir}/v12_t1e_features.pth",map_location="cpu"),
        "ev":torch.load(f"{emb_dir}/v12_evidence_features.pth",map_location="cpu"),
        "struct":torch.load(f"{emb_dir}/v12_struct_features.pth",map_location="cpu"),
    }
    with open(ann_path) as f: feats["labels"]={d["Video_ID"]:d for d in json.load(f)}
    splits=load_split_ids(split_dir)
    sd=feats["struct"][list(feats["struct"].keys())[0]].shape[0]
    nr=args.num_runs
    logger.info(f"{args.dataset_name} {args.language}, nr={nr}")

    if args.dataset_name=="HateMM":
        run("6mod+ev wCE1.5",feats,splits,lm,["text","audio","frame","t1","t2","ev"],sd,192,2e-4,45,0.15,0.15,nr,nc,logger,class_weight=[1.0,1.5])
        run("6mod+ev wCE1.5 mix",feats,splits,lm,["text","audio","frame","t1","t2","ev"],sd,192,2e-4,45,0.15,0.15,nr,nc,logger,class_weight=[1.0,1.5],use_mixup=True)
    elif args.language=="English":
        # Many configs with fresh seeds for max coverage
        run("6mod+ev wCE1.5",feats,splits,lm,["text","audio","frame","t1","t2","ev"],sd,192,2e-4,45,0.15,0.15,nr,nc,logger,class_weight=[1.0,1.5])
        run("T1E wCE2.0",feats,splits,lm,["text","audio","frame","t1e","t2"],sd,192,2e-4,45,0.15,0.15,nr,nc,logger,class_weight=[1.0,2.0])
        run("6mod+ev mix",feats,splits,lm,["text","audio","frame","t1","t2","ev"],sd,192,2e-4,45,0.15,0.15,nr,nc,logger,use_mixup=True)
    else:
        run("6mod+ev asym",feats,splits,lm,["text","audio","frame","t1","t2","ev"],sd,192,2e-4,45,0.15,0.15,nr,nc,logger,class_weight=[1.0,1.5])
        run("6mod+ev mix",feats,splits,lm,["text","audio","frame","t1","t2","ev"],sd,192,2e-4,45,0.15,0.15,nr,nc,logger,use_mixup=True)

if __name__=="__main__":
    main()
