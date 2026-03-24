"""Ablation on BEST seed + BEST config per dataset. Remove one component at a time."""
import csv,json,os,random,copy,sys
import numpy as np
import torch,torch.nn as nn,torch.nn.functional as F,torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score,confusion_matrix
from sklearn.covariance import LedoitWolf
from torch.utils.data import DataLoader,Dataset
import warnings;warnings.filterwarnings("ignore")
device="cuda"

class DS(Dataset):
    def __init__(self,vids,feats,lm,mk):
        self.vids=vids;self.f=feats;self.lm=lm;self.mk=mk
    def __len__(self):return len(self.vids)
    def __getitem__(self,i):
        v=self.vids[i];out={k:self.f[k][v] for k in self.mk}
        out["struct"]=self.f["struct"][v]
        out["label"]=torch.tensor(self.lm[self.f["labels"][v]["Label"]],dtype=torch.long)
        return out
def collate_fn(b):return{k:torch.stack([x[k] for x in b]) for k in b[0]}

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

def spca_whiten(tr,va,te,r=32):
    mean=tr.mean(dim=0,keepdim=True);c=(tr-mean).numpy()
    lw=LedoitWolf().fit(c);cov=torch.tensor(lw.covariance_,dtype=torch.float32)
    U,S,V=torch.svd(cov)
    if r and r<U.size(1):U=U[:,:r];S=S[:r];V=V[:,:r]
    W=U@torch.diag(1.0/torch.sqrt(S+1e-6))
    return(F.normalize((tr-mean)@W,dim=1),F.normalize((va-mean)@W,dim=1),F.normalize((te-mean)@W,dim=1))

def knn_logits(qe,be,bl,k=10,nc=2,temperature=0.05):
    qn=F.normalize(qe,dim=1);bn=F.normalize(be,dim=1)
    sim=torch.mm(qn,bn.t());ts2,ti=sim.topk(k,dim=1)
    tl=bl[ti];w=F.softmax(ts2/temperature,dim=1)
    out=torch.zeros(qe.size(0),nc)
    for c in range(nc):out[:,c]=(w*(tl==c).float()).sum(dim=1)
    return out.numpy()

def csls_knn(query,bank,bl,k=10,nc=2,temperature=0.05,hub_k=10):
    qn=F.normalize(query,dim=1);bn=F.normalize(bank,dim=1)
    sim=torch.mm(qn,bn.t())
    bh=sim.topk(min(hub_k,sim.size(0)),dim=0).values.mean(dim=0)
    cs=2*sim-bh.unsqueeze(0)
    ts2,ti=cs.topk(k,dim=1);tl=bl[ti]
    w=F.softmax(ts2/temperature,dim=1)
    out=torch.zeros(query.size(0),nc)
    for c in range(nc):out[:,c]=(w*(tl==c).float()).sum(dim=1)
    return out.numpy()

def run_one(feats,cur,lm,mk,sd,nc,seed,cw_list,wtype,ktype,kk,kt,ka):
    """Train model, optionally apply whitening+kNN, return ACC/MF1/MP/MR."""
    torch.manual_seed(seed);random.seed(seed);np.random.seed(seed)
    trd=DS(cur["train"],feats,lm,mk);vd=DS(cur["valid"],feats,lm,mk);ted=DS(cur["test"],feats,lm,mk)
    trl=DataLoader(trd,32,True,collate_fn=collate_fn);vl=DataLoader(vd,64,False,collate_fn=collate_fn)
    tel=DataLoader(ted,64,False,collate_fn=collate_fn)
    trl_ns=DataLoader(DS(cur["train"],feats,lm,mk),64,False,collate_fn=collate_fn)
    model=Fusion(mk,hidden=192,sd=sd,nc=nc,drop=0.15,md=0.15).to(device)
    ema=copy.deepcopy(model)
    opt=optim.AdamW(model.parameters(),lr=2e-4,weight_decay=0.02)
    crit=nn.CrossEntropyLoss(weight=torch.tensor(cw_list,dtype=torch.float).to(device),label_smoothing=0.03)
    ep=45;ts_t=ep*len(trl);ws=5*len(trl);sch=cw(opt,ws,ts_t)
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
    tp,tl_arr,tla=get_pl(ema,trl_ns);vp,vl_arr,vla=get_pl(ema,vl);tep,tel_arr,tela=get_pl(ema,tel)
    blt=torch.tensor(tla)
    if ka>0:
        if wtype=="spca_r32":tr_w,va_w,te_w=spca_whiten(tp,vp,tep,r=32)
        else:tr_w,va_w,te_w=tp,vp,tep
        if ktype=="csls":knn_t=csls_knn(te_w,tr_w,blt,k=kk,nc=nc,temperature=kt)
        else:knn_t=knn_logits(te_w,tr_w,blt,k=kk,nc=nc,temperature=kt)
        fl=(1-ka)*tel_arr+ka*knn_t
    else:
        fl=tel_arr
    preds=np.argmax(fl,axis=1)
    acc=accuracy_score(tela,preds);mf1=f1_score(tela,preds,average='macro')
    mp=precision_score(tela,preds,average='macro');mr=recall_score(tela,preds,average='macro')
    cm=confusion_matrix(tela,preds)
    return acc,mf1,mp,mr,cm

def main():
    dataset=sys.argv[1];language=sys.argv[2] if len(sys.argv)>2 else "English"
    if dataset=="HateMM":
        emb_dir="./embeddings/HateMM";ann_path="./datasets/HateMM/annotation(new).json"
        split_dir="./datasets/HateMM/splits";lm={"Non Hate":0,"Hate":1}
        SEED=505042;BW="spca_r32";BKT="csls";BK=10;BT=0.05;BA=0.15
    elif language=="English":
        emb_dir="./embeddings/Multihateclip/English"
        ann_path="./datasets/Multihateclip/English/annotation(new).json"
        split_dir="./datasets/Multihateclip/English/splits";lm={"Normal":0,"Offensive":1,"Hateful":1}
        SEED=501042;BW="none";BKT="cosine";BK=10;BT=0.02;BA=0.35
    else:
        emb_dir="./embeddings/Multihateclip/Chinese"
        ann_path="./datasets/Multihateclip/Chinese/annotation(new).json"
        split_dir="./datasets/Multihateclip/Chinese/splits";lm={"Normal":0,"Offensive":1,"Hateful":1}
        SEED=530042;BW="spca_r32";BKT="cosine";BK=10;BT=0.02;BA=0.45

    nc=2;CW=[1.0,1.5]
    feats={
        "text":torch.load(f"{emb_dir}/text_features.pth",map_location="cpu"),
        "audio":torch.load(f"{emb_dir}/wavlm_audio_features.pth",map_location="cpu"),
        "frame":torch.load(f"{emb_dir}/frame_features.pth",map_location="cpu"),
        "t1":torch.load(f"{emb_dir}/v12_t1_features.pth",map_location="cpu"),
        "t2":torch.load(f"{emb_dir}/v12_t2_features.pth",map_location="cpu"),
        "ev":torch.load(f"{emb_dir}/v12_evidence_features.pth",map_location="cpu"),
        "struct":torch.load(f"{emb_dir}/v12_struct_features.pth",map_location="cpu"),
        "v9t1":torch.load(f"{emb_dir}/v9_t1_features.pth",map_location="cpu"),
        "v9t2":torch.load(f"{emb_dir}/v9_t2_features.pth",map_location="cpu"),
        "v9struct":torch.load(f"{emb_dir}/v9_struct_features.pth",map_location="cpu"),
    }
    with open(ann_path) as f:feats["labels"]={d["Video_ID"]:d for d in json.load(f)}
    splits=load_split_ids(split_dir)
    mk6=["text","audio","frame","t1","t2","ev"]
    fk=mk6+["struct"]
    common=set.intersection(*[set(feats[k].keys()) for k in fk])&set(feats["labels"].keys())
    cur={s:[v for v in splits[s] if v in common] for s in splits}
    sd=feats["struct"][list(feats["struct"].keys())[0]].shape[0]

    # v9 common
    feats_v9=dict(feats);feats_v9["struct"]=feats["v9struct"]
    v9mk=["text","audio","frame","v9t1","v9t2"]
    v9fk=v9mk+["struct"]
    v9c=set.intersection(*[set(feats_v9[k].keys()) for k in v9fk])&set(feats_v9["labels"].keys())
    v9cur={s:[v for v in splits[s] if v in v9c] for s in splits}
    v9sd=feats["v9struct"][list(feats["v9struct"].keys())[0]].shape[0]

    def p(name,r):
        acc,mf1,mp,mr,cm=r
        print(f"  {name:<45} ACC={acc:.4f}  M-F1={mf1:.4f}  M-P={mp:.4f}  M-R={mr:.4f}  CM={cm.tolist()}")
        sys.stdout.flush()

    print(f"\n{'='*80}")
    print(f"  {dataset} {language} — Best Seed Ablation (seed={SEED})")
    print(f"{'='*80}")

    # Full model with retrieval
    print(f"\n--- Full Model ---")
    p("Full (6mod+whiten+kNN)", run_one(feats,cur,lm,mk6,sd,nc,SEED,CW,BW,BKT,BK,BT,BA))

    # Staged ablation
    print(f"\n--- Staged Ablation ---")
    p("Raw Fusion (text+audio+frame)", run_one(feats,cur,lm,["text","audio","frame"],sd,nc,SEED,CW,"none","none",0,0,0))
    p("+ Evidence", run_one(feats,cur,lm,["text","audio","frame","ev"],sd,nc,SEED,CW,"none","none",0,0,0))
    p("+ Relational meaning", run_one(feats,cur,lm,["text","audio","frame","ev","t1"],sd,nc,SEED,CW,"none","none",0,0,0))
    p("+ Alt. appraisals (=6mod)", run_one(feats,cur,lm,mk6,sd,nc,SEED,CW,"none","none",0,0,0))
    p("+ kNN (no whitening)", run_one(feats,cur,lm,mk6,sd,nc,SEED,CW,"none","cosine",BK,BT,BA))
    p("+ Whiten+kNN (full)", run_one(feats,cur,lm,mk6,sd,nc,SEED,CW,BW,BKT,BK,BT,BA))

    # Field ablation (with retrieval)
    print(f"\n--- Field Ablation (with retrieval) ---")
    p("Full (6mod+retrieval)", run_one(feats,cur,lm,mk6,sd,nc,SEED,CW,BW,BKT,BK,BT,BA))
    p("- Evidence", run_one(feats,cur,lm,["text","audio","frame","t1","t2"],sd,nc,SEED,CW,BW,BKT,BK,BT,BA))
    p("- Relational meaning", run_one(feats,cur,lm,["text","audio","frame","t2","ev"],sd,nc,SEED,CW,BW,BKT,BK,BT,BA))
    p("- Alt. appraisals", run_one(feats,cur,lm,["text","audio","frame","t1","ev"],sd,nc,SEED,CW,BW,BKT,BK,BT,BA))
    p("- All LLM fields", run_one(feats,cur,lm,["text","audio","frame"],sd,nc,SEED,CW,BW,BKT,BK,BT,BA))

    # Field ablation (no retrieval)
    print(f"\n--- Field Ablation (no retrieval) ---")
    p("Full 6mod (no retrieval)", run_one(feats,cur,lm,mk6,sd,nc,SEED,CW,"none","none",0,0,0))
    p("- Evidence (no retr)", run_one(feats,cur,lm,["text","audio","frame","t1","t2"],sd,nc,SEED,CW,"none","none",0,0,0))
    p("- Relational meaning (no retr)", run_one(feats,cur,lm,["text","audio","frame","t2","ev"],sd,nc,SEED,CW,"none","none",0,0,0))
    p("- Alt. appraisals (no retr)", run_one(feats,cur,lm,["text","audio","frame","t1","ev"],sd,nc,SEED,CW,"none","none",0,0,0))

    # 1-call vs 2-call
    print(f"\n--- 1-Call vs 2-Call ---")
    p("1-Call (v9)", run_one(feats_v9,v9cur,lm,v9mk,v9sd,nc,SEED,CW,"none","none",0,0,0))
    p("2-Call no evidence", run_one(feats,cur,lm,["text","audio","frame","t1","t2"],sd,nc,SEED,CW,"none","none",0,0,0))
    p("2-Call full", run_one(feats,cur,lm,mk6,sd,nc,SEED,CW,"none","none",0,0,0))

    # Retrieval comparison
    print(f"\n--- Retrieval Comparison ---")
    p("Head only (no kNN)", run_one(feats,cur,lm,mk6,sd,nc,SEED,CW,"none","none",0,0,0))
    p("+ Raw kNN (no whiten)", run_one(feats,cur,lm,mk6,sd,nc,SEED,CW,"none","cosine",BK,BT,BA))
    p("+ Whiten+kNN", run_one(feats,cur,lm,mk6,sd,nc,SEED,CW,BW,BKT,BK,BT,BA))

    print(f"\n{'='*80}")
    print(f"  DONE — {dataset} {language}")
    print(f"{'='*80}")

if __name__=="__main__":
    main()
