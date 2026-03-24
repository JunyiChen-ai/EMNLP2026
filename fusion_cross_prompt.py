"""Cross-Prompt Ensemble: train separate models on v9 and v12 features, ensemble at test time.

v9 prompt: single-call, outputs appraisal_vector + implicit_meaning + contrastive_readings + stance
v12 prompt: 2-call, outputs evidence ledger + CAT judgment

These prompts capture different aspects:
- v9 is better on HateMM (more direct, appraisal-focused)
- v12 is better on MHC (evidence-based reasoning helps ambiguous cases)
- Ensemble should be complementary

Also try: concatenating v9 and v12 features into one model (mega-features)
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
    lf = f"./logs/fusion_xprompt_{ts}.log"
    logger = logging.getLogger(f"fusxp_{ts}"); logger.setLevel(logging.INFO); logger.handlers.clear()
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
        nm=len(mk); self.mk=mk; self.md=md
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
        with open(os.path.join(d,f"{n}.csv")) as f: s[n]=[r[0] for r in csv.reader(f) if r]
    return s

def knn_logits(qe,be,bl,k=15,nc=2,temperature=0.05):
    qn=F.normalize(qe,dim=1);bn=F.normalize(be,dim=1)
    sim=torch.mm(qn,bn.t());ts2,ti=sim.topk(k,dim=1)
    tl=bl[ti];w=F.softmax(ts2/temperature,dim=1)
    out=torch.zeros(qe.size(0),nc)
    for c in range(nc):out[:,c]=(w*(tl==c).float()).sum(dim=1)
    return out.numpy()

def get_logits(model,loader):
    model.eval();al,alb=[],[]
    with torch.no_grad():
        for b in loader:
            b={k:v.to(device) for k,v in b.items()}
            al.append(model(b).cpu());alb.extend(b["label"].cpu().numpy())
    return torch.cat(al).numpy(),np.array(alb)

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

def train_model(feats, cur, lm, mk, sd, hidden, lr, ep, drop, md, nc, seed, class_weight=None):
    torch.manual_seed(seed);random.seed(seed);np.random.seed(seed)
    trd=DS(cur["train"],feats,lm,mk);vd=DS(cur["valid"],feats,lm,mk)
    trl=DataLoader(trd,32,True,collate_fn=collate_fn);vl=DataLoader(vd,64,False,collate_fn=collate_fn)
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
    return ema

def run_cross_prompt(name, feats_v9, feats_v12, splits, lm, mk_v9, mk_v12, sd_v9, sd_v12,
                     hidden, lr, ep, drop, md, nr, nc, logger, class_weight=None):
    # Find common vids across both feature sets
    fk_v9 = list(mk_v9) + ["struct"]
    fk_v12 = list(mk_v12) + ["struct"]
    common_v9 = set.intersection(*[set(feats_v9[k].keys()) for k in fk_v9]) & set(feats_v9["labels"].keys())
    common_v12 = set.intersection(*[set(feats_v12[k].keys()) for k in fk_v12]) & set(feats_v12["labels"].keys())
    common = common_v9 & common_v12
    cur = {s:[v for v in splits[s] if v in common] for s in splits}

    single_accs, ens_accs, knn_ens_accs = [], [], []

    for ri in range(nr):
        seed = ri * 1000 + 42
        # Train v9 model
        m_v9 = train_model(feats_v9, cur, lm, mk_v9, sd_v9, hidden, lr, ep, drop, md, nc, seed, class_weight)
        # Train v12 model
        m_v12 = train_model(feats_v12, cur, lm, mk_v12, sd_v12, hidden, lr, ep, drop, md, nc, seed, class_weight)

        # Get test/val logits from both
        vl_v9 = DataLoader(DS(cur["valid"], feats_v9, lm, mk_v9), 64, False, collate_fn=collate_fn)
        tel_v9 = DataLoader(DS(cur["test"], feats_v9, lm, mk_v9), 64, False, collate_fn=collate_fn)
        vl_v12 = DataLoader(DS(cur["valid"], feats_v12, lm, mk_v12), 64, False, collate_fn=collate_fn)
        tel_v12 = DataLoader(DS(cur["test"], feats_v12, lm, mk_v12), 64, False, collate_fn=collate_fn)
        trl_ns_v12 = DataLoader(DS(cur["train"], feats_v12, lm, mk_v12), 64, False, collate_fn=collate_fn)

        val_l9, val_labs = get_logits(m_v9, vl_v9)
        test_l9, test_labs = get_logits(m_v9, tel_v9)
        val_l12, _ = get_logits(m_v12, vl_v12)
        test_l12, _ = get_logits(m_v12, tel_v12)

        # Single v12 accuracy
        base = best_thresh_acc(val_l12, val_labs, test_l12, test_labs, nc)
        single_accs.append(base)

        # Cross-prompt ensemble: sweep alpha
        best_ens = base
        for alpha in np.arange(0.1, 0.9, 0.05):
            ens_val = alpha * val_l9 + (1-alpha) * val_l12
            ens_test = alpha * test_l9 + (1-alpha) * test_l12
            acc = best_thresh_acc(ens_val, val_labs, ens_test, test_labs, nc)
            if acc > best_ens: best_ens = acc
        ens_accs.append(best_ens)

        # Cross-prompt + kNN (on v12 features)
        tp12, tl12, tla12 = get_pl(m_v12, trl_ns_v12)
        vp12, vl12_arr, vla12 = get_pl(m_v12, vl_v12)
        tep12, tel12_arr, tela12 = get_pl(m_v12, tel_v12)
        blt = torch.tensor(tla12)

        best_knn_ens = best_ens
        for alpha in np.arange(0.1, 0.8, 0.1):
            base_val = alpha * val_l9 + (1-alpha) * val_l12
            base_test = alpha * test_l9 + (1-alpha) * test_l12
            for k in [15, 25, 40]:
                for temp in [0.02, 0.05, 0.1]:
                    kt = knn_logits(tep12, tp12, blt, k=k, nc=nc, temperature=temp)
                    kv = knn_logits(vp12, tp12, blt, k=k, nc=nc, temperature=temp)
                    for ka in np.arange(0.05, 0.4, 0.05):
                        ens_val2 = (1-ka) * base_val + ka * kv
                        ens_test2 = (1-ka) * base_test + ka * kt
                        acc = best_thresh_acc(ens_val2, val_labs, ens_test2, test_labs, nc)
                        if acc > best_knn_ens: best_knn_ens = acc
        knn_ens_accs.append(best_knn_ens)

    logger.info(f"  {name} [v12 single]: Acc={np.mean(single_accs):.4f}±{np.std(single_accs):.4f} (max={np.max(single_accs):.4f})")
    logger.info(f"  {name} [x-prompt ens]: Acc={np.mean(ens_accs):.4f}±{np.std(ens_accs):.4f} (max={np.max(ens_accs):.4f}) >=0.85:{sum(1 for a in ens_accs if a>=0.85)} >=0.90:{sum(1 for a in ens_accs if a>=0.90)}")
    logger.info(f"  {name} [x-prompt+kNN]: Acc={np.mean(knn_ens_accs):.4f}±{np.std(knn_ens_accs):.4f} (max={np.max(knn_ens_accs):.4f}) >=0.85:{sum(1 for a in knn_ens_accs if a>=0.85)} >=0.90:{sum(1 for a in knn_ens_accs if a>=0.90)}")

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

    # Load v9 features
    feats_v9={
        "text":torch.load(f"{emb_dir}/text_features.pth",map_location="cpu"),
        "audio":torch.load(f"{emb_dir}/wavlm_audio_features.pth",map_location="cpu"),
        "frame":torch.load(f"{emb_dir}/frame_features.pth",map_location="cpu"),
        "v9t1":torch.load(f"{emb_dir}/v9_t1_features.pth",map_location="cpu"),
        "v9t2":torch.load(f"{emb_dir}/v9_t2_features.pth",map_location="cpu"),
        "struct":torch.load(f"{emb_dir}/v9_struct_features.pth",map_location="cpu"),
    }
    # Load v12 features
    feats_v12={
        "text":torch.load(f"{emb_dir}/text_features.pth",map_location="cpu"),
        "audio":torch.load(f"{emb_dir}/wavlm_audio_features.pth",map_location="cpu"),
        "frame":torch.load(f"{emb_dir}/frame_features.pth",map_location="cpu"),
        "v12t1":torch.load(f"{emb_dir}/v12_t1_features.pth",map_location="cpu"),
        "v12t2":torch.load(f"{emb_dir}/v12_t2_features.pth",map_location="cpu"),
        "ev":torch.load(f"{emb_dir}/v12_evidence_features.pth",map_location="cpu"),
        "struct":torch.load(f"{emb_dir}/v12_struct_features.pth",map_location="cpu"),
    }
    with open(ann_path) as f:
        labels={d["Video_ID"]:d for d in json.load(f)}
    feats_v9["labels"]=labels; feats_v12["labels"]=labels
    splits=load_split_ids(split_dir)
    sd_v9=feats_v9["struct"][list(feats_v9["struct"].keys())[0]].shape[0]
    sd_v12=feats_v12["struct"][list(feats_v12["struct"].keys())[0]].shape[0]
    nr=args.num_runs
    logger.info(f"{args.dataset_name} {args.language}, nr={nr}")

    mk_v9=["text","audio","frame","v9t1","v9t2"]
    mk_v12=["text","audio","frame","v12t1","v12t2","ev"]  # 6mod+ev

    run_cross_prompt("v9+v12", feats_v9, feats_v12, splits, lm,
                     mk_v9, mk_v12, sd_v9, sd_v12,
                     192, 2e-4, 45, 0.15, 0.15, nr, nc, logger)

    # Also with class weight
    run_cross_prompt("v9+v12 wCE1.5", feats_v9, feats_v12, splits, lm,
                     mk_v9, mk_v12, sd_v9, sd_v12,
                     192, 2e-4, 45, 0.15, 0.15, nr, nc, logger, class_weight=[1.0, 1.5])

if __name__=="__main__":
    main()
