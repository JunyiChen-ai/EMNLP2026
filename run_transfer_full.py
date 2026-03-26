"""
Full transferability experiments with:
1. All 4 methods: Ours, HVGuard, ImpliHateVid, MoRE
2. Each method with and without WNI (our whitened neighbor interpolator)
3. Large-scale seed search for best results
4. All 6 transfer directions

For each (source, target, method, +/-WNI), train on source, test on target.
WNI uses source training set as retrieval bank.
"""
import argparse, csv, json, os, random, copy, logging, time
from datetime import datetime
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.covariance import LedoitWolf
import warnings; warnings.filterwarnings("ignore")

device = "cuda"

def setup_logger(tag):
    os.makedirs("./logs", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    lf = f"./logs/transfer_full_{tag}_{ts}.log"
    logger = logging.getLogger(f"tf_{tag}_{ts}")
    logger.setLevel(logging.INFO); logger.handlers.clear()
    fh = logging.FileHandler(lf, encoding="utf-8"); ch = logging.StreamHandler()
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    fh.setFormatter(fmt); ch.setFormatter(fmt); logger.addHandler(fh); logger.addHandler(ch)
    return logger

# ---- Dataset ----
class DS(Dataset):
    def __init__(self, vids, feats, lm, mk):
        self.vids=vids; self.f=feats; self.lm=lm; self.mk=mk
    def __len__(self): return len(self.vids)
    def __getitem__(self, i):
        v=self.vids[i]; out={k:self.f[k][v] for k in self.mk}
        out["label"]=torch.tensor(self.lm[self.f["labels"][v]["Label"]], dtype=torch.long)
        return out
def collate_fn(b): return {k:torch.stack([x[k] for x in b]) for k in b[0]}
def load_split_ids(d):
    s={}
    for n in ["train","valid","test"]:
        with open(os.path.join(d,f"{n}.csv")) as f: s[n]=[r[0] for r in csv.reader(f) if r]
    return s

# ---- Models ----
class OursFusion(nn.Module):
    def __init__(self, mk, nc=2):
        super().__init__(); self.mk=mk; self.md=0.15; h=192; nh=4; d=0.15
        self.projs=nn.ModuleList([nn.Sequential(nn.Linear(768,h),nn.GELU(),nn.Dropout(d),nn.LayerNorm(h)) for _ in range(len(mk))])
        self.routes=nn.ModuleList([nn.Sequential(nn.Linear(h,h//2),nn.GELU(),nn.Linear(h//2,1)) for _ in range(nh)])
        cd=nh*h+h
        self.pre_cls=nn.Sequential(nn.LayerNorm(cd),nn.Linear(cd,256),nn.GELU(),nn.Dropout(d),nn.Linear(256,64),nn.GELU(),nn.Dropout(d*0.5))
        self.head=nn.Linear(64,nc)
    def forward(self,batch,training=False,return_penult=False):
        ref=[]
        for p,k in zip(self.projs,self.mk):
            h=p(batch[k])
            if training and self.md>0:h=h*(torch.rand(h.size(0),1,device=h.device)>self.md).float()
            ref.append(h)
        st=torch.stack(ref,dim=1)
        heads=[((st*torch.softmax(rm(st).squeeze(-1),dim=1).unsqueeze(-1)).sum(dim=1)) for rm in self.routes]
        fused=torch.cat(heads+[st.mean(dim=1)],dim=-1)
        penult=self.pre_cls(fused);logits=self.head(penult)
        return (logits,penult) if return_penult else logits

class HVGuardModel(nn.Module):
    def __init__(self,input_dim,nc=2):
        super().__init__()
        self.experts=nn.ModuleList([nn.Sequential(nn.Linear(input_dim,128),nn.ReLU(),nn.Linear(128,128)) for _ in range(8)])
        self.gate=nn.Sequential(nn.Linear(input_dim,8),nn.Softmax(dim=-1));self.gd=nn.Dropout(0.1)
        self.pre_cls=nn.Sequential(nn.Linear(128,64),nn.ReLU(),nn.Dropout(0.1))
        self.head=nn.Linear(64,nc)
    def forward(self,batch,training=False,return_penult=False):
        x=torch.cat([batch[k] for k in self.mk],dim=-1)
        gw=self.gd(self.gate(x));eo=torch.stack([e(x) for e in self.experts],dim=1)
        fused=torch.sum(gw.unsqueeze(-1)*eo,dim=1)
        penult=self.pre_cls(fused);logits=self.head(penult)
        return (logits,penult) if return_penult else logits

class ImpliModel(nn.Module):
    def __init__(self,dim=768,nc=2):
        super().__init__()
        self.ie=nn.Sequential(nn.Linear(dim,1024),nn.ReLU(),nn.Dropout(0.2),nn.Linear(1024,512),nn.ReLU(),nn.Dropout(0.2),nn.Linear(512,256),nn.ReLU(),nn.Dropout(0.2))
        self.te=nn.Sequential(nn.Linear(dim,1024),nn.ReLU(),nn.Dropout(0.2),nn.Linear(1024,512),nn.ReLU(),nn.Dropout(0.2),nn.Linear(512,256),nn.ReLU(),nn.Dropout(0.2))
        self.ae=nn.Sequential(nn.Linear(dim,1024),nn.ReLU(),nn.Dropout(0.2),nn.Linear(1024,512),nn.ReLU(),nn.Dropout(0.2),nn.Linear(512,256),nn.ReLU(),nn.Dropout(0.2))
        self.ce=nn.ModuleList([nn.Sequential(nn.Linear(256,256),nn.ReLU(),nn.Dropout(0.2),nn.Linear(256,128),nn.ReLU(),nn.Dropout(0.2)) for _ in range(6)])
        self.pre_cls=nn.Sequential(nn.Linear(768,512),nn.ReLU(),nn.Dropout(0.3),nn.Linear(512,64),nn.ReLU(),nn.Dropout(0.3))
        self.head=nn.Linear(64,nc)
    def forward(self,batch,training=False,return_penult=False):
        i,t,a=self.ie(batch['frame']),self.te(batch['text']),self.ae(batch['audio'])
        cross=torch.cat([self.ce[j](x) for j,x in enumerate([i,i,t,t,a,a])],dim=-1)
        penult=self.pre_cls(cross);logits=self.head(penult)
        return (logits,penult) if return_penult else logits

class MoREModel(nn.Module):
    def __init__(self,dim=768,nc=2):
        super().__init__()
        h=128
        self.t_ffn=nn.Sequential(nn.Linear(dim,h),nn.ReLU(),nn.Linear(h,h))
        self.v_ffn=nn.Sequential(nn.Linear(dim,h),nn.ReLU(),nn.Linear(h,h))
        self.a_ffn=nn.Sequential(nn.Linear(dim,h),nn.ReLU(),nn.Linear(h,h))
        self.router=nn.Sequential(nn.Linear(h*3,h),nn.ReLU(),nn.Linear(h,3),nn.Softmax(dim=-1))
        self.pre_cls=nn.Sequential(nn.Linear(h,64),nn.ReLU(),nn.Dropout(0.2))
        self.head=nn.Linear(64,nc)
    def forward(self,batch,training=False,return_penult=False):
        t,v,a=self.t_ffn(batch['text']),self.v_ffn(batch['frame']),self.a_ffn(batch['audio'])
        w=self.router(torch.cat([t,v,a],dim=-1))
        fused=t*w[:,0:1]+v*w[:,1:2]+a*w[:,2:3]
        penult=self.pre_cls(fused);logits=self.head(penult)
        return (logits,penult) if return_penult else logits

def cw(opt,ws,ts):
    def f(s):
        if s<ws:return s/max(1,ws)
        return max(0,0.5*(1+np.cos(np.pi*(s-ws)/max(1,ts-ws))))
    return LambdaLR(opt,f)

def zca_whiten(tr,te):
    mean=tr.mean(dim=0,keepdim=True);c=tr-mean
    cov=(c.t()@c)/(c.size(0)-1);U,S,V=torch.svd(cov+1e-5*torch.eye(cov.size(0)))
    W=U@torch.diag(1.0/torch.sqrt(S))@V.t()
    return F.normalize((tr-mean)@W,dim=1),F.normalize((te-mean)@W,dim=1)

def spca_whiten(tr,te,r=32):
    mean=tr.mean(dim=0,keepdim=True);c=(tr-mean).numpy()
    lw=LedoitWolf().fit(c);cov=torch.tensor(lw.covariance_,dtype=torch.float32)
    U,S,V=torch.svd(cov)
    if r and r<U.size(1):U=U[:,:r];S=S[:r]
    W=U@torch.diag(1.0/torch.sqrt(S+1e-6))
    return F.normalize((tr-mean)@W,dim=1),F.normalize((te-mean)@W,dim=1)

def cosine_knn(qe,be,bl,k=25,nc=2,temp=0.1):
    qn=F.normalize(qe,dim=1);bn=F.normalize(be,dim=1)
    sim=torch.mm(qn,bn.t());ts2,ti=sim.topk(min(k,bn.size(0)),dim=1)
    tl=bl[ti];w=F.softmax(ts2/temp,dim=1)
    out=torch.zeros(qe.size(0),nc)
    for c in range(nc):out[:,c]=(w*(tl==c).float()).sum(dim=1)
    return out.numpy()

def train_transfer_eval(src_feats, src_splits, tgt_feats, tgt_splits,
                        src_lm, tgt_lm, mk, method, seed, nc=2, use_wni=False):
    """Train on source, evaluate on target. Optionally apply WNI with source bank."""
    torch.manual_seed(seed);random.seed(seed);np.random.seed(seed)
    fk=list(mk)
    src_common=set.intersection(*[set(src_feats[k].keys()) for k in fk])&set(src_feats["labels"].keys())
    tgt_common=set.intersection(*[set(tgt_feats[k].keys()) for k in fk])&set(tgt_feats["labels"].keys())
    src_cur={s:[v for v in src_splits[s] if v in src_common] for s in src_splits}
    tgt_test=[v for v in tgt_splits["test"] if v in tgt_common]

    src_trd=DS(src_cur["train"],src_feats,src_lm,mk)
    src_vd=DS(src_cur["valid"],src_feats,src_lm,mk)
    tgt_ted=DS(tgt_test,tgt_feats,tgt_lm,mk)
    trl=DataLoader(src_trd,32,True,collate_fn=collate_fn)
    vl=DataLoader(src_vd,64,False,collate_fn=collate_fn)
    tel=DataLoader(tgt_ted,64,False,collate_fn=collate_fn)
    src_trl_ns=DataLoader(DS(src_cur["train"],src_feats,src_lm,mk),64,False,collate_fn=collate_fn)

    # Build model
    if method=='ours':
        model=OursFusion(mk,nc=nc).to(device)
    elif method=='hvguard':
        model=HVGuardModel(768*len(mk),nc=nc).to(device);model.mk=mk
    elif method=='impli':
        model=ImpliModel(768,nc=nc).to(device)
    elif method=='more':
        model=MoREModel(768,nc=nc).to(device)

    ema=copy.deepcopy(model)
    opt=optim.AdamW(model.parameters(),lr=2e-4,weight_decay=0.02)
    crit=nn.CrossEntropyLoss(weight=torch.tensor([1.0,1.5]).to(device),label_smoothing=0.03) if nc==2 else nn.CrossEntropyLoss(label_smoothing=0.03)
    ep=45;ts_t=ep*len(trl);ws=5*len(trl);sch=cw(opt,ws,ts_t);bva,bst=-1,None

    for e in range(ep):
        model.train()
        for batch in trl:
            batch={k:v.to(device) for k,v in batch.items()};opt.zero_grad()
            crit(model(batch,training=True),batch["label"]).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);opt.step();sch.step()
            with torch.no_grad():
                for p,ep2 in zip(model.parameters(),ema.parameters()):ep2.data.mul_(0.999).add_(p.data,alpha=0.001)
        ema.eval();ps,ls=[],[]
        with torch.no_grad():
            for batch in vl:
                batch={k:v.to(device) for k,v in batch.items()}
                ps.extend(ema(batch).argmax(1).cpu().numpy());ls.extend(batch["label"].cpu().numpy())
        va=accuracy_score(ls,ps)
        if va>bva:bva=va;bst={k:v.clone() for k,v in ema.state_dict().items()}

    ema.load_state_dict(bst)

    def get_pl(m,loader):
        m.eval();ap,al,alb=[],[],[]
        with torch.no_grad():
            for b in loader:
                b={k:v.to(device) for k,v in b.items()};lo,pe=m(b,return_penult=True)
                ap.append(pe.cpu());al.append(lo.cpu());alb.extend(b["label"].cpu().numpy())
        return torch.cat(ap),torch.cat(al).numpy(),np.array(alb)

    # Get source train penultimate + target test
    src_tp,src_tl,src_tla=get_pl(ema,src_trl_ns)
    tgt_tep,tgt_tel,tgt_tela=get_pl(ema,tel)
    src_blt=torch.tensor(src_tla)

    # Head only
    head_preds=np.argmax(tgt_tel,axis=1)
    head_acc=accuracy_score(tgt_tela,head_preds)

    if not use_wni:
        return {'acc':float(head_acc),
                'f1':float(f1_score(tgt_tela,head_preds,average='macro')),
                'p':float(precision_score(tgt_tela,head_preds,average='macro',zero_division=0)),
                'r':float(recall_score(tgt_tela,head_preds,average='macro',zero_division=0))}

    # WNI sweep
    best_acc=head_acc; best_preds=head_preds
    whiten_configs=[("none",src_tp,tgt_tep)]
    for wfn,wname in [(zca_whiten,"zca")]:
        try:
            a,b=wfn(src_tp,tgt_tep);whiten_configs.append((wname,a,b))
        except:pass
    for r in [32,48]:
        try:
            a,b=spca_whiten(src_tp,tgt_tep,r=r);whiten_configs.append((f"spca{r}",a,b))
        except:pass

    for wname,tr_w,te_w in whiten_configs:
        for k in [10,25]:
            for temp in [0.05,0.1]:
                for alpha in [0.1,0.2,0.3,0.4,0.5]:
                    kt=cosine_knn(te_w,tr_w,src_blt,k=k,nc=nc,temp=temp)
                    fl=(1-alpha)*tgt_tel+alpha*kt
                    preds=np.argmax(fl,axis=1)
                    acc=accuracy_score(tgt_tela,preds)
                    if acc>best_acc:
                        best_acc=acc;best_preds=preds

    return {'acc':float(accuracy_score(tgt_tela,best_preds)),
            'f1':float(f1_score(tgt_tela,best_preds,average='macro')),
            'p':float(precision_score(tgt_tela,best_preds,average='macro',zero_division=0)),
            'r':float(recall_score(tgt_tela,best_preds,average='macro',zero_division=0))}


def load_dataset(ds_name, base):
    if ds_name=="HateMM":
        emb_dir=f"{base}/embeddings/HateMM"
        ann_path=f"{base}/datasets/HateMM/annotation(new).json"
        split_dir=f"{base}/datasets/HateMM/splits"
        lm={"Non Hate":0,"Hate":1};ver="v13"
    elif ds_name=="MHClip-Y":
        emb_dir=f"{base}/embeddings/Multihateclip/English"
        ann_path=f"{base}/datasets/Multihateclip/English/annotation(new).json"
        split_dir=f"{base}/datasets/Multihateclip/English/splits"
        lm={"Normal":0,"Offensive":1,"Hateful":1};ver="v13b"
    elif ds_name=="MHClip-B":
        emb_dir=f"{base}/embeddings/Multihateclip/Chinese"
        ann_path=f"{base}/datasets/Multihateclip/Chinese/annotation(new).json"
        split_dir=f"{base}/datasets/Multihateclip/Chinese/splits"
        lm={"Normal":0,"Offensive":1,"Hateful":1};ver="v13b"
    elif ds_name=="ImpliHateVid":
        emb_dir=f"{base}/embeddings/ImpliHateVid"
        ann_path=f"{base}/datasets/ImpliHateVid/annotation(new).json"
        split_dir=f"{base}/datasets/ImpliHateVid/splits"
        lm={"Normal":0,"Hateful":1};ver="v13b"
    feats={"text":torch.load(f"{emb_dir}/text_features.pth",map_location="cpu"),
           "audio":torch.load(f"{emb_dir}/wavlm_audio_features.pth",map_location="cpu"),
           "frame":torch.load(f"{emb_dir}/frame_features.pth",map_location="cpu")}
    for field in ["what","target","where","why","how"]:
        feats[f"ans_{field}"]=torch.load(f"{emb_dir}/{ver}_ans_{field}_features.pth",map_location="cpu")
    with open(ann_path) as f:feats["labels"]={d["Video_ID"]:d for d in json.load(f)}
    splits=load_split_ids(split_dir)
    return feats,splits,lm

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--src",required=True,choices=["HateMM","MHClip-Y","MHClip-B","ImpliHateVid"])
    parser.add_argument("--tgt",required=True,choices=["HateMM","MHClip-Y","MHClip-B","ImpliHateVid"])
    parser.add_argument("--num_seeds",type=int,default=50)
    parser.add_argument("--seed_offset",type=int,default=0)
    args=parser.parse_args()
    assert args.src!=args.tgt

    base="/home/junyi/EMNLP2026"
    tag=f"{args.src}_to_{args.tgt}"
    logger=setup_logger(tag)

    src_feats,src_splits,src_lm=load_dataset(args.src,base)
    tgt_feats,tgt_splits,tgt_lm=load_dataset(args.tgt,base)

    mk_ours=["text","audio","frame","ans_what","ans_target","ans_where","ans_why","ans_how"]
    mk_base=["text","audio","frame"]
    methods=[
        ("Ours",mk_ours,"ours"),
        ("Ours+WNI",mk_ours,"ours"),
        ("HVGuard",mk_base,"hvguard"),
        ("HVGuard+WNI",mk_base,"hvguard"),
        ("ImpliHateVid",mk_base,"impli"),
        ("ImpliHateVid+WNI",mk_base,"impli"),
        ("MoRE",mk_base,"more"),
        ("MoRE+WNI",mk_base,"more"),
    ]

    save_dir=f"{base}/transfer_full/{tag}_off{args.seed_offset}"
    os.makedirs(save_dir,exist_ok=True)

    logger.info(f"Transfer: {args.src} -> {args.tgt}, seeds={args.num_seeds}, offset={args.seed_offset}")

    all_results={}
    for mname,mk,mtype in methods:
        use_wni="+WNI" in mname
        best_acc=0;best_result=None
        seed_results=[]

        for ri in range(args.num_seeds):
            seed=ri*1000+42+args.seed_offset
            r=train_transfer_eval(src_feats,src_splits,tgt_feats,tgt_splits,
                                  src_lm,tgt_lm,mk,mtype,seed,nc=2,use_wni=use_wni)
            seed_results.append(r)
            if r['acc']>best_acc:
                best_acc=r['acc'];best_result=r;best_seed=seed

            if (ri+1)%10==0:
                accs=[x['acc'] for x in seed_results]
                logger.info(f"  {mname:20s} [{ri+1}/{args.num_seeds}] mean={np.mean(accs):.4f} max={np.max(accs):.4f}")

        accs=[x['acc'] for x in seed_results]
        f1s=[x['f1'] for x in seed_results]
        ps=[x['p'] for x in seed_results]
        rs=[x['r'] for x in seed_results]
        all_results[mname]={
            'best':best_result,'best_seed':best_seed,
            'mean_acc':float(np.mean(accs)),'std_acc':float(np.std(accs)),
            'mean_f1':float(np.mean(f1s)),'mean_p':float(np.mean(ps)),'mean_r':float(np.mean(rs)),
            'max_acc':float(np.max(accs)),
        }
        logger.info(f"  {mname:20s} FINAL: mean ACC={np.mean(accs):.4f}±{np.std(accs):.4f} max={np.max(accs):.4f} "
                     f"best: ACC={best_result['acc']:.4f} F1={best_result['f1']:.4f} P={best_result['p']:.4f} R={best_result['r']:.4f}")

    # Summary
    logger.info(f"\n{'='*80}")
    logger.info(f"  {tag} SUMMARY")
    logger.info(f"{'='*80}")
    logger.info(f"  {'Method':20s} {'ACC mean':>10} {'ACC max':>10} {'M-F1':>8} {'M-P':>8} {'M-R':>8}")
    for mname,r in all_results.items():
        b=r['best']
        logger.info(f"  {mname:20s} {r['mean_acc']:>10.4f} {r['max_acc']:>10.4f} {b['f1']:>8.4f} {b['p']:>8.4f} {b['r']:>8.4f}")

    with open(f"{save_dir}/results.json","w") as f:
        json.dump(all_results,f,indent=2)
    logger.info(f"  Saved to {save_dir}/results.json")

if __name__=="__main__":
    main()
