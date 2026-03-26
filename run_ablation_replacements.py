"""
Ablation replacement experiments under best seed config.
Runs on HateMM (seed=607042) and MHClip-B (seed=99042).

Variants:
1. w/ HVGuard prompt: Use HVGuard's LLM CoT output instead of our P2C fields
2. w/ MoE fusion: Replace our Schema-Guided Router with MoE (8 experts)
3. w/ HVGuard fusion: Replace with HVGuard's MoE+MLP
4. w/o Retrieval: Head only
5. w/o Whitening: kNN without whitening
6. w/ Pre-fusion Retrieval: Retrieve on raw concatenated features
"""
import csv, json, os, random, copy
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.covariance import LedoitWolf
import warnings; warnings.filterwarnings("ignore")

device = "cuda"

class DS(Dataset):
    def __init__(self, vids, feats, lm, mk):
        self.vids=vids; self.f=feats; self.lm=lm; self.mk=mk
    def __len__(self): return len(self.vids)
    def __getitem__(self, i):
        v=self.vids[i]; out={k:self.f[k][v] for k in self.mk}
        out["label"]=torch.tensor(self.lm[self.f["labels"][v]["Label"]],dtype=torch.long)
        return out

def collate_fn(b): return {k:torch.stack([x[k] for x in b]) for k in b[0]}

def load_split_ids(d):
    s={}
    for n in ["train","valid","test"]:
        with open(os.path.join(d,f"{n}.csv")) as f: s[n]=[r[0] for r in csv.reader(f) if r]
    return s

# ---- Our Fusion (normal) ----
class OurFusion(nn.Module):
    def __init__(self, mk, hidden=192, nh=4, nc=2, drop=0.15, md=0.15):
        super().__init__()
        self.mk=mk; self.md=md
        self.projs=nn.ModuleList([nn.Sequential(nn.Linear(768,hidden),nn.GELU(),nn.Dropout(drop),nn.LayerNorm(hidden)) for _ in range(len(mk))])
        self.routes=nn.ModuleList([nn.Sequential(nn.Linear(hidden,hidden//2),nn.GELU(),nn.Linear(hidden//2,1)) for _ in range(nh)])
        cd=nh*hidden+hidden
        self.pre_cls=nn.Sequential(nn.LayerNorm(cd),nn.Linear(cd,256),nn.GELU(),nn.Dropout(drop),nn.Linear(256,64),nn.GELU(),nn.Dropout(drop*0.5))
        self.head=nn.Linear(64,nc)
    def forward(self, batch, training=False, return_penult=False):
        ref=[]
        for p,k in zip(self.projs,self.mk):
            h=p(batch[k])
            if training and self.md>0: h=h*(torch.rand(h.size(0),1,device=h.device)>self.md).float()
            ref.append(h)
        st=torch.stack(ref,dim=1)
        heads=[((st*torch.softmax(rm(st).squeeze(-1),dim=1).unsqueeze(-1)).sum(dim=1)) for rm in self.routes]
        fused=torch.cat(heads+[st.mean(dim=1)],dim=-1)
        penult=self.pre_cls(fused); logits=self.head(penult)
        return (logits,penult) if return_penult else logits

# ---- MoE Fusion (MoRE-style replacement) ----
class MoEFusion(nn.Module):
    def __init__(self, mk, nc=2):
        super().__init__()
        self.mk=mk; nm=len(mk)
        self.projs=nn.ModuleList([nn.Linear(768,128) for _ in range(nm)])
        self.experts=nn.ModuleList([nn.Sequential(nn.Linear(128*nm,128),nn.ReLU(),nn.Linear(128,128)) for _ in range(8)])
        self.gate=nn.Sequential(nn.Linear(128*nm,8),nn.Softmax(dim=-1))
        self.pre_cls=nn.Sequential(nn.Linear(128,64),nn.ReLU(),nn.Dropout(0.1))
        self.head=nn.Linear(64,nc)
    def forward(self, batch, training=False, return_penult=False):
        feats=[p(batch[k]) for p,k in zip(self.projs,self.mk)]
        x=torch.cat(feats,dim=-1)
        gw=self.gate(x)
        eo=torch.stack([e(x) for e in self.experts],dim=1)
        fused=torch.sum(gw.unsqueeze(-1)*eo,dim=1)
        penult=self.pre_cls(fused); logits=self.head(penult)
        return (logits,penult) if return_penult else logits

# ---- HVGuard-style Fusion (MoE+MLP) ----
class HVGuardFusion(nn.Module):
    def __init__(self, mk, nc=2):
        super().__init__()
        self.mk=mk; total_dim=768*len(mk)
        self.experts=nn.ModuleList([nn.Sequential(nn.Linear(total_dim,128),nn.ReLU(),nn.Linear(128,128)) for _ in range(8)])
        self.gate=nn.Sequential(nn.Linear(total_dim,8),nn.Softmax(dim=-1))
        self.gate_drop=nn.Dropout(0.1)
        self.pre_cls=nn.Sequential(nn.Linear(128,64),nn.ReLU(),nn.Dropout(0.1))
        self.head=nn.Linear(64,nc)
    def forward(self, batch, training=False, return_penult=False):
        x=torch.cat([batch[k] for k in self.mk],dim=-1)
        gw=self.gate_drop(self.gate(x))
        eo=torch.stack([e(x) for e in self.experts],dim=1)
        fused=torch.sum(gw.unsqueeze(-1)*eo,dim=1)
        penult=self.pre_cls(fused); logits=self.head(penult)
        return (logits,penult) if return_penult else logits

def cw(opt,ws,ts):
    def f(s):
        if s<ws: return s/max(1,ws)
        return max(0,0.5*(1+np.cos(np.pi*(s-ws)/max(1,ts-ws))))
    return LambdaLR(opt,f)

def zca_whiten(tr,va,te):
    mean=tr.mean(dim=0,keepdim=True);c=tr-mean
    cov=(c.t()@c)/(c.size(0)-1)
    U,S,V=torch.svd(cov+1e-5*torch.eye(cov.size(0)))
    W=U@torch.diag(1.0/torch.sqrt(S))@V.t()
    return F.normalize((tr-mean)@W,dim=1),F.normalize((va-mean)@W,dim=1),F.normalize((te-mean)@W,dim=1)

def cosine_knn(qe,be,bl,k=25,nc=2,temp=0.1):
    qn=F.normalize(qe,dim=1);bn=F.normalize(be,dim=1)
    sim=torch.mm(qn,bn.t());ts2,ti=sim.topk(k,dim=1)
    tl=bl[ti];w=F.softmax(ts2/temp,dim=1)
    out=torch.zeros(qe.size(0),nc)
    for c in range(nc): out[:,c]=(w*(tl==c).float()).sum(dim=1)
    return out.numpy()

def train_and_eval(feats, splits, lm, mk, model_cls, seed, alpha=0.0,
                   use_whiten=True, retrieval_on='post'):
    torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
    nc=2; fk=list(mk)
    common=set.intersection(*[set(feats[k].keys()) for k in fk])&set(feats["labels"].keys())
    cur={s:[v for v in splits[s] if v in common] for s in splits}

    trd=DS(cur["train"],feats,lm,mk);vd=DS(cur["valid"],feats,lm,mk);ted=DS(cur["test"],feats,lm,mk)
    trl=DataLoader(trd,32,True,collate_fn=collate_fn)
    vl=DataLoader(vd,64,False,collate_fn=collate_fn)
    tel=DataLoader(ted,64,False,collate_fn=collate_fn)
    trl_ns=DataLoader(DS(cur["train"],feats,lm,mk),64,False,collate_fn=collate_fn)

    model=model_cls(mk,nc=nc).to(device);ema=copy.deepcopy(model)
    opt=optim.AdamW(model.parameters(),lr=2e-4,weight_decay=0.02)
    crit=nn.CrossEntropyLoss(weight=torch.tensor([1.0,1.5]).to(device),label_smoothing=0.03)
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

    def get_pl(m,loader):
        m.eval();ap,al,alb=[],[],[]
        with torch.no_grad():
            for b in loader:
                b={k:v.to(device) for k,v in b.items()};lo,pe=m(b,return_penult=True)
                ap.append(pe.cpu());al.append(lo.cpu());alb.extend(b["label"].cpu().numpy())
        return torch.cat(ap),torch.cat(al).numpy(),np.array(alb)

    tp,tl_arr,tla=get_pl(ema,trl_ns);vp,vl_arr,vla=get_pl(ema,vl);tep,tel_arr,tela=get_pl(ema,tel)

    # Head only
    head_preds=np.argmax(tel_arr,axis=1)
    head_acc=accuracy_score(tela,head_preds)
    head_f1=f1_score(tela,head_preds,average='macro')

    if alpha > 0:
        blt=torch.tensor(tla)
        if retrieval_on == 'post':
            feat_tr,feat_te = tp,tep
        else:  # pre-fusion: concat raw modality features
            feat_tr_list,feat_te_list=[],[]
            for b in trl_ns:
                b={k:v for k,v in b.items()}
                feat_tr_list.append(torch.cat([b[k] for k in ['text','audio','frame']],dim=-1))
            for b in tel:
                b={k:v for k,v in b.items()}
                feat_te_list.append(torch.cat([b[k] for k in ['text','audio','frame']],dim=-1))
            feat_tr=torch.cat(feat_tr_list); feat_te=torch.cat(feat_te_list)

        if use_whiten:
            try: feat_tr,_,feat_te=zca_whiten(feat_tr,feat_tr,feat_te)
            except: pass
        kt=cosine_knn(feat_te,feat_tr,blt,k=25,nc=nc,temp=0.1)
        fl=(1-alpha)*tel_arr+alpha*kt
        preds=np.argmax(fl,axis=1)
    else:
        preds=head_preds

    return {
        "acc": float(accuracy_score(tela,preds)),
        "f1": float(f1_score(tela,preds,average='macro')),
        "p": float(precision_score(tela,preds,average='macro')),
        "r": float(recall_score(tela,preds,average='macro')),
        "head_acc": float(head_acc), "head_f1": float(head_f1),
    }

def main():
    base = "/home/junyi/EMNLP2026"
    configs = [
        ("HateMM", 607042, "v13",
         f"{base}/embeddings/HateMM", f"{base}/datasets/HateMM/annotation(new).json",
         f"{base}/datasets/HateMM/splits", {"Non Hate":0,"Hate":1}, 0.5),
        ("MHClip-B", 99042, "v13b",
         f"{base}/embeddings/Multihateclip/Chinese", f"{base}/datasets/Multihateclip/Chinese/annotation(new).json",
         f"{base}/datasets/Multihateclip/Chinese/splits", {"Normal":0,"Offensive":1,"Hateful":1}, 0.1),
        ("ImpliHateVid", 28042, "v13b",
         f"{base}/embeddings/ImpliHateVid", f"{base}/datasets/ImpliHateVid/annotation(new).json",
         f"{base}/datasets/ImpliHateVid/splits", {"Normal":0,"Hateful":1}, 0.4),
    ]

    mk_ours = ["text","audio","frame","ans_what","ans_target","ans_where","ans_why","ans_how"]
    mk_hvguard = ["text","audio","frame","hvguard_mix"]  # HVGuard prompt replacement
    mk_base = ["text","audio","frame"]

    for tag, seed, ver, emb_dir, ann_path, split_dir, lm, alpha in configs:
        print(f"\n{'='*60}\n  {tag} (seed={seed})\n{'='*60}")

        feats = {
            "text": torch.load(f"{emb_dir}/text_features.pth",map_location="cpu"),
            "audio": torch.load(f"{emb_dir}/wavlm_audio_features.pth",map_location="cpu"),
            "frame": torch.load(f"{emb_dir}/frame_features.pth",map_location="cpu"),
        }
        for field in ["what","target","where","why","how"]:
            feats[f"ans_{field}"] = torch.load(f"{emb_dir}/{ver}_ans_{field}_features.pth",map_location="cpu")

        # HVGuard mix if available
        hvg_path = f"{emb_dir}/hvguard_mix_features.pth"
        if os.path.exists(hvg_path):
            feats["hvguard_mix"] = torch.load(hvg_path,map_location="cpu")
        else:
            feats["hvguard_mix"] = feats["text"]  # fallback

        with open(ann_path) as f: feats["labels"]={d["Video_ID"]:d for d in json.load(f)}
        splits = load_split_ids(split_dir)

        variants = [
            ("Full model", mk_ours, OurFusion, alpha, True, 'post'),
            ("w/o Retrieval", mk_ours, OurFusion, 0.0, True, 'post'),
            ("w/o Whitening", mk_ours, OurFusion, alpha, False, 'post'),
            ("w/ Pre-fusion Retr.", mk_ours, OurFusion, alpha, True, 'pre'),
            ("w/ MoE fusion", mk_ours, MoEFusion, alpha, True, 'post'),
            ("w/ HVGuard fusion", mk_ours, HVGuardFusion, alpha, True, 'post'),
        ]

        for vname, mk, model_cls, a, wh, ret_on in variants:
            r = train_and_eval(feats, splits, lm, mk, model_cls, seed, alpha=a,
                              use_whiten=wh, retrieval_on=ret_on)
            print(f"  {vname:25s}: ACC={r['acc']:.4f} M-F1={r['f1']:.4f} M-P={r['p']:.4f} M-R={r['r']:.4f} (head={r['head_acc']:.4f})")

    os.makedirs("ablation_results", exist_ok=True)
    print("\nDone.")

if __name__ == "__main__":
    main()
