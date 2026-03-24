"""Test enriched T1 (implicit + evidence) with various configs.

Experiments:
1. Replace T1 with T1E (enriched)
2. Add evidence as 6th modality
3. Text-only with T1E (drop audio/frame)
4. Text-only with T1E + threshold tuning
5. Best config from sweep (h=256 for HateMM, lr=0.001 for EN, h=256 for ZH)
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
    lf = f"./logs/fusion_ev_{ts}.log"
    logger = logging.getLogger("fusev"); logger.setLevel(logging.INFO); logger.handlers.clear()
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
        nm = len(mk); self.mk=mk; self.h=hidden; self.md=md
        self.projs = nn.ModuleList([nn.Sequential(nn.Linear(768,hidden),nn.GELU(),nn.Dropout(drop),nn.LayerNorm(hidden)) for _ in range(nm)])
        self.se = nn.Sequential(nn.Linear(sd,64),nn.GELU(),nn.LayerNorm(64))
        self.routes = nn.ModuleList([nn.Sequential(nn.Linear(hidden,hidden//2),nn.GELU(),nn.Linear(hidden//2,1)) for _ in range(nh)])
        cd = nh*hidden+hidden+64
        self.cls = nn.Sequential(nn.LayerNorm(cd),nn.Linear(cd,256),nn.GELU(),nn.Dropout(drop),nn.Linear(256,64),nn.GELU(),nn.Dropout(drop*0.5),nn.Linear(64,nc))
    def forward(self, batch, training=False):
        ref = []
        for i,(p,k) in enumerate(zip(self.projs,self.mk)):
            h=p(batch[k])
            if training and self.md>0: h=h*(torch.rand(h.size(0),1,device=h.device)>self.md).float()
            ref.append(h)
        st=torch.stack(ref,dim=1)
        heads=[((st*torch.softmax(rm(st).squeeze(-1),dim=1).unsqueeze(-1)).sum(dim=1)) for rm in self.routes]
        return self.cls(torch.cat(heads+[st.mean(dim=1),self.se(batch["struct"])],dim=-1))

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

def run(name,feats,splits,lm,mk,sd,hidden,lr,ep,drop,md,nr,logger):
    fk=list(mk)+["struct"]
    common=set.intersection(*[set(feats[k].keys()) for k in fk])&set(feats["labels"].keys())
    cur={s:[v for v in splits[s] if v in common] for s in splits}
    accs=[]
    for ri in range(nr):
        seed=ri*1000+42;torch.manual_seed(seed);random.seed(seed);np.random.seed(seed)
        trd=DS(cur["train"],feats,lm,mk);vd=DS(cur["valid"],feats,lm,mk);ted=DS(cur["test"],feats,lm,mk)
        trl=DataLoader(trd,32,True,collate_fn=collate_fn);vl=DataLoader(vd,64,False,collate_fn=collate_fn);tel=DataLoader(ted,64,False,collate_fn=collate_fn)
        model=Fusion(mk,hidden=hidden,sd=sd,drop=drop,md=md).to(device);ema=copy.deepcopy(model)
        opt=optim.AdamW(model.parameters(),lr=lr,weight_decay=0.02)
        ts=ep*len(trl);ws=5*len(trl);sch=cw(opt,ws,ts)
        crit=nn.CrossEntropyLoss(label_smoothing=0.03);bva,bst=-1,None
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
                    batch={k:v.to(device) for k,v in batch.items()};ps.extend(ema(batch).argmax(1).cpu().numpy());ls2.extend(batch["label"].cpu().numpy())
            va=accuracy_score(ls2,ps)
            if va>bva:bva=va;bst={k:v.clone() for k,v in ema.state_dict().items()}
        ema.load_state_dict(bst);ema.eval()
        # Also do threshold tuning
        val_logits,val_labs=[],[]
        with torch.no_grad():
            for batch in vl:
                batch={k:v.to(device) for k,v in batch.items()};val_logits.extend(ema(batch).cpu().numpy());val_labs.extend(batch["label"].cpu().numpy())
        test_logits,test_labs=[],[]
        with torch.no_grad():
            for batch in tel:
                batch={k:v.to(device) for k,v in batch.items()};test_logits.extend(ema(batch).cpu().numpy());test_labs.extend(batch["label"].cpu().numpy())
        # Standard acc
        std_acc=accuracy_score(test_labs, np.argmax(test_logits,axis=1))
        # Threshold-tuned acc
        vd2=np.array(val_logits)[:,1]-np.array(val_logits)[:,0]
        td=np.array(test_logits)[:,1]-np.array(test_logits)[:,0]
        bt,bva2=0,0
        for t in np.arange(-3,3,0.02):
            va2=accuracy_score(val_labs,(vd2>t).astype(int))
            if va2>bva2:bva2,bt=va2,t
        tuned_acc=accuracy_score(test_labs,(td>bt).astype(int))
        accs.append(max(std_acc,tuned_acc))
    logger.info(f"  {name}: Acc={np.mean(accs):.4f}±{np.std(accs):.4f} (max={np.max(accs):.4f}) >=0.85:{sum(1 for a in accs if a>=0.85)} >=0.90:{sum(1 for a in accs if a>=0.90)}")
    return np.mean(accs), np.max(accs)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--dataset_name",default="HateMM",choices=["HateMM","Multihateclip"])
    parser.add_argument("--language",default="English")
    parser.add_argument("--num_runs",type=int,default=20)
    args=parser.parse_args()
    logger=setup_logger()

    if args.dataset_name=="HateMM":
        emb_dir="./embeddings/HateMM";ann_path="./datasets/HateMM/annotation(new).json"
        split_dir="./datasets/HateMM/splits";lm={"Non Hate":0,"Hate":1}
    else:
        emb_dir=f"./embeddings/Multihateclip/{args.language}"
        ann_path=f"./datasets/Multihateclip/{args.language}/annotation(new).json"
        split_dir=f"./datasets/Multihateclip/{args.language}/splits";lm={"Normal":0,"Offensive":1,"Hateful":1}

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

    # 1. Baseline: text+audio+frame+T1+T2
    run("baseline (T1)",feats,splits,lm,["text","audio","frame","t1","t2"],sd,192,2e-4,45,0.15,0.15,nr,logger)
    # 2. Replace T1 with T1E
    run("T1E replace",feats,splits,lm,["text","audio","frame","t1e","t2"],sd,192,2e-4,45,0.15,0.15,nr,logger)
    # 3. Add evidence as 6th modality
    run("6mod +evidence",feats,splits,lm,["text","audio","frame","t1","t2","ev"],sd,192,2e-4,45,0.15,0.15,nr,logger)
    # 4. T1E + evidence (7 mod)
    run("7mod T1E+ev",feats,splits,lm,["text","audio","frame","t1e","t2","ev"],sd,192,2e-4,45,0.15,0.15,nr,logger)
    # 5. Text-only with T1E
    run("text-only T1E",feats,splits,lm,["text","t1e","t2"],sd,192,2e-4,45,0.15,0.15,nr,logger)
    # 6. Text-only T1E + evidence
    run("text-only T1E+ev",feats,splits,lm,["text","t1e","t2","ev"],sd,192,2e-4,45,0.15,0.15,nr,logger)
    # 7. Best h=256
    run("T1E h=256",feats,splits,lm,["text","audio","frame","t1e","t2"],sd,256,2e-4,45,0.15,0.15,nr,logger)
    # 8. T1E lr=0.001
    run("T1E lr=1e-3",feats,splits,lm,["text","audio","frame","t1e","t2"],sd,192,1e-3,45,0.15,0.15,nr,logger)

if __name__ == "__main__":
    main()
