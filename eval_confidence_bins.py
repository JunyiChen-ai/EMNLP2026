"""Confidence-binned retrieval analysis: show kNN gain by classifier confidence quartile."""
import csv, json, os, random, copy
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from sklearn.metrics import accuracy_score
from sklearn.covariance import LedoitWolf
from torch.utils.data import DataLoader, Dataset
import warnings; warnings.filterwarnings("ignore")

device = "cuda"

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

def shrinkage_pca_whiten(train_z,val_z,test_z,r=None):
    mean=train_z.mean(dim=0,keepdim=True)
    centered=(train_z-mean).numpy()
    lw=LedoitWolf().fit(centered)
    cov=torch.tensor(lw.covariance_,dtype=torch.float32)
    U,S,V=torch.svd(cov)
    if r and r<U.size(1):U=U[:,:r];S=S[:r];V=V[:,:r]
    W=U@torch.diag(1.0/torch.sqrt(S+1e-6))
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

configs = {
    "HateMM": {
        "seed": 505042, "whiten": "spca_r32", "knn_type": "csls",
        "k": 10, "temp": 0.05, "alpha": 0.15,
        "emb_dir": "./embeddings/HateMM",
        "ann_path": "./datasets/HateMM/annotation(new).json",
        "split_dir": "./datasets/HateMM/splits",
        "lm": {"Non Hate": 0, "Hate": 1},
    },
    "EN_MHC": {
        "seed": 501042, "whiten": "none", "knn_type": "cosine",
        "k": 10, "temp": 0.02, "alpha": 0.35,
        "emb_dir": "./embeddings/Multihateclip/English",
        "ann_path": "./datasets/Multihateclip/English/annotation(new).json",
        "split_dir": "./datasets/Multihateclip/English/splits",
        "lm": {"Normal": 0, "Offensive": 1, "Hateful": 1},
    },
    "ZH_MHC": {
        "seed": 530042, "whiten": "spca_r32", "knn_type": "cosine",
        "k": 10, "temp": 0.02, "alpha": 0.45,
        "emb_dir": "./embeddings/Multihateclip/Chinese",
        "ann_path": "./datasets/Multihateclip/Chinese/annotation(new).json",
        "split_dir": "./datasets/Multihateclip/Chinese/splits",
        "lm": {"Normal": 0, "Offensive": 1, "Hateful": 1},
    },
}

mk = ["text","audio","frame","t1","t2","ev"]
nc = 2

for dname, cfg in configs.items():
    print(f"\n{'='*60}")
    print(f"  {dname} — Confidence-Binned Retrieval Analysis")
    print(f"{'='*60}")

    feats = {
        "text": torch.load(f"{cfg['emb_dir']}/text_features.pth", map_location="cpu"),
        "audio": torch.load(f"{cfg['emb_dir']}/wavlm_audio_features.pth", map_location="cpu"),
        "frame": torch.load(f"{cfg['emb_dir']}/frame_features.pth", map_location="cpu"),
        "t1": torch.load(f"{cfg['emb_dir']}/v12_t1_features.pth", map_location="cpu"),
        "t2": torch.load(f"{cfg['emb_dir']}/v12_t2_features.pth", map_location="cpu"),
        "ev": torch.load(f"{cfg['emb_dir']}/v12_evidence_features.pth", map_location="cpu"),
        "struct": torch.load(f"{cfg['emb_dir']}/v12_struct_features.pth", map_location="cpu"),
    }
    with open(cfg['ann_path']) as f:
        feats["labels"] = {d["Video_ID"]: d for d in json.load(f)}
    splits = load_split_ids(cfg['split_dir'])
    sd = feats["struct"][list(feats["struct"].keys())[0]].shape[0]
    fk = list(mk) + ["struct"]
    common = set.intersection(*[set(feats[k].keys()) for k in fk]) & set(feats["labels"].keys())
    cur = {s: [v for v in splits[s] if v in common] for s in splits}

    seed = cfg["seed"]
    torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
    trd=DS(cur["train"],feats,cfg["lm"],mk);vd=DS(cur["valid"],feats,cfg["lm"],mk);ted=DS(cur["test"],feats,cfg["lm"],mk)
    trl=DataLoader(trd,32,True,collate_fn=collate_fn)
    vl=DataLoader(vd,64,False,collate_fn=collate_fn)
    tel=DataLoader(ted,64,False,collate_fn=collate_fn)
    trl_ns=DataLoader(DS(cur["train"],feats,cfg["lm"],mk),64,False,collate_fn=collate_fn)

    model=Fusion(mk,hidden=192,sd=sd,nc=nc,drop=0.15,md=0.15).to(device)
    ema=copy.deepcopy(model)
    opt=optim.AdamW(model.parameters(),lr=2e-4,weight_decay=0.02)
    crit=nn.CrossEntropyLoss(weight=torch.tensor([1.0,1.5],dtype=torch.float).to(device),label_smoothing=0.03)
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

    # Whitening
    if cfg["whiten"]=="none":
        tr_w,va_w,te_w=tp,vp,tep
    elif cfg["whiten"].startswith("spca_r"):
        r=int(cfg["whiten"].split("r")[1])
        tr_w,va_w,te_w=shrinkage_pca_whiten(tp,vp,tep,r=r)

    # kNN
    if cfg["knn_type"]=="cosine":
        kt=knn_logits(te_w,tr_w,blt,k=cfg["k"],nc=nc,temperature=cfg["temp"])
    else:
        kt=csls_knn(te_w,tr_w,blt,k=cfg["k"],nc=nc,temperature=cfg["temp"])

    alpha=cfg["alpha"]
    blended=((1-alpha)*tel_arr+alpha*kt)

    # Head predictions and confidence
    head_probs=F.softmax(torch.tensor(tel_arr),dim=1).numpy()
    head_conf=head_probs.max(axis=1)  # max probability as confidence
    head_preds=np.argmax(tel_arr,axis=1)
    blend_preds=np.argmax(blended,axis=1)

    # Entropy as alternative confidence measure
    head_entropy=-(head_probs*np.log(head_probs+1e-8)).sum(axis=1)

    # Bin by confidence quartiles
    quartiles=np.percentile(head_conf,[25,50,75])
    bin_names=["Q1 (lowest)","Q2","Q3","Q4 (highest)"]
    bin_edges=[0,quartiles[0],quartiles[1],quartiles[2],1.01]

    print(f"\n  Confidence quartile edges: {quartiles}")
    print(f"\n  {'Bin':<16} {'N':>4} {'Head ACC':>10} {'Blend ACC':>10} {'Gain':>8} {'Head FN':>8} {'Rescued':>8}")
    print(f"  {'-'*66}")

    total_fn_head=0; total_rescued=0
    for bi in range(4):
        mask=(head_conf>=bin_edges[bi])&(head_conf<bin_edges[bi+1])
        n=mask.sum()
        if n==0:continue
        h_acc=accuracy_score(tela[mask],head_preds[mask])
        b_acc=accuracy_score(tela[mask],blend_preds[mask])
        gain=b_acc-h_acc

        # FN analysis: hateful samples predicted as non-hateful by head
        fn_head=((head_preds[mask]==0)&(tela[mask]==1)).sum()
        fn_blend=((blend_preds[mask]==0)&(tela[mask]==1)).sum()
        rescued=fn_head-fn_blend
        total_fn_head+=fn_head; total_rescued+=rescued

        print(f"  {bin_names[bi]:<16} {n:>4} {h_acc:>10.4f} {b_acc:>10.4f} {gain:>+8.4f} {fn_head:>8} {rescued:>+8}")

    print(f"  {'-'*66}")
    print(f"  {'Total':<16} {len(tela):>4} {accuracy_score(tela,head_preds):>10.4f} {accuracy_score(tela,blend_preds):>10.4f} {accuracy_score(tela,blend_preds)-accuracy_score(tela,head_preds):>+8.4f} {total_fn_head:>8} {total_rescued:>+8}")

    # Also show: samples where retrieval FLIPPED the prediction
    flipped_correct=((head_preds!=tela)&(blend_preds==tela)).sum()
    flipped_wrong=((head_preds==tela)&(blend_preds!=tela)).sum()
    print(f"\n  Retrieval flips: {flipped_correct} correct→correct, {flipped_wrong} correct→wrong")
    print(f"  Net flipped: +{flipped_correct} rescued, -{flipped_wrong} broken = net {flipped_correct-flipped_wrong}")
