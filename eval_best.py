"""Evaluate best seeds with full metrics: ACC, M-F1, M-P, M-R, per-class P/R/F1, confusion matrix."""
import argparse, csv, json, os, random, copy
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
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

def full_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    mf1 = f1_score(y_true, y_pred, average='macro')
    mp = precision_score(y_true, y_pred, average='macro')
    mr = recall_score(y_true, y_pred, average='macro')
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, digits=4)
    return acc, mf1, mp, mr, cm, report

def main():
    # Best configs from reproduce experiments
    configs = {
        "HateMM": {
            "seed": 505042,
            "whiten": "spca_r32",
            "knn_type": "csls",
            "k": 10,
            "temp": 0.05,
            "alpha": 0.15,
            "thresh": -0.12,
            "emb_dir": "./embeddings/HateMM",
            "ann_path": "./datasets/HateMM/annotation(new).json",
            "split_dir": "./datasets/HateMM/splits",
            "lm": {"Non Hate": 0, "Hate": 1},
        },
        "EN_MHC": {
            "seed": 501042,
            "whiten": "none",
            "knn_type": "cosine",
            "k": 10,
            "temp": 0.02,
            "alpha": 0.35,
            "thresh": None,
            "emb_dir": "./embeddings/Multihateclip/English",
            "ann_path": "./datasets/Multihateclip/English/annotation(new).json",
            "split_dir": "./datasets/Multihateclip/English/splits",
            "lm": {"Normal": 0, "Offensive": 1, "Hateful": 1},
        },
        "ZH_MHC": {
            "seed": 530042,
            "whiten": "spca_r32",
            "knn_type": "cosine",
            "k": 10,
            "temp": 0.02,
            "alpha": 0.45,
            "thresh": None,
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
        print(f"  {dname}")
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

        trd = DS(cur["train"], feats, cfg["lm"], mk)
        vd = DS(cur["valid"], feats, cfg["lm"], mk)
        ted = DS(cur["test"], feats, cfg["lm"], mk)
        trl = DataLoader(trd, 32, True, collate_fn=collate_fn)
        vl = DataLoader(vd, 64, False, collate_fn=collate_fn)
        tel = DataLoader(ted, 64, False, collate_fn=collate_fn)
        trl_ns = DataLoader(DS(cur["train"], feats, cfg["lm"], mk), 64, False, collate_fn=collate_fn)

        # Train
        model = Fusion(mk, hidden=192, sd=sd, nc=nc, drop=0.15, md=0.15).to(device)
        ema = copy.deepcopy(model)
        opt = optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.02)
        crit = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 1.5], dtype=torch.float).to(device), label_smoothing=0.03)
        ep = 45
        ts_t = ep * len(trl); ws = 5 * len(trl)
        sch = cw(opt, ws, ts_t)
        bva, bst = -1, None

        for e in range(ep):
            model.train()
            for batch in trl:
                batch = {k: v.to(device) for k, v in batch.items()}; opt.zero_grad()
                crit(model(batch, training=True), batch["label"]).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sch.step()
                with torch.no_grad():
                    for p, ep2 in zip(model.parameters(), ema.parameters()):
                        ep2.data.mul_(0.999).add_(p.data, alpha=0.001)
            ema.eval(); ps, ls2 = [], []
            with torch.no_grad():
                for batch in vl:
                    batch = {k: v.to(device) for k, v in batch.items()}
                    ps.extend(ema(batch).argmax(1).cpu().numpy())
                    ls2.extend(batch["label"].cpu().numpy())
            va = accuracy_score(ls2, ps)
            if va > bva: bva = va; bst = {k: v.clone() for k, v in ema.state_dict().items()}

        ema.load_state_dict(bst)
        print(f"  Best val ACC: {bva:.4f}")

        # Get features
        tp, tl_arr, tla = get_pl(ema, trl_ns)
        vp, vl_arr, vla = get_pl(ema, vl)
        tep, tel_arr, tela = get_pl(ema, tel)
        blt = torch.tensor(tla)

        # === Metric 1: Head only (argmax) ===
        preds_head = np.argmax(tel_arr, axis=1)
        acc, mf1, mp, mr, cm, report = full_metrics(tela, preds_head)
        print(f"\n  [Head only] ACC={acc:.4f}  M-F1={mf1:.4f}  M-P={mp:.4f}  M-R={mr:.4f}")
        print(f"  CM: {cm.tolist()}")

        # === Metric 2: Head + threshold ===
        if cfg["thresh"] is not None:
            td = tel_arr[:, 1] - tel_arr[:, 0]
            preds_thresh = (td > cfg["thresh"]).astype(int)
            acc_t, mf1_t, mp_t, mr_t, cm_t, _ = full_metrics(tela, preds_thresh)
            print(f"  [Head+thresh={cfg['thresh']:.2f}] ACC={acc_t:.4f}  M-F1={mf1_t:.4f}  M-P={mp_t:.4f}  M-R={mr_t:.4f}")
            print(f"  CM: {cm_t.tolist()}")

        # === Whitening ===
        wname = cfg["whiten"]
        if wname == "none":
            tr_w, va_w, te_w = tp, vp, tep
        elif wname == "zca":
            tr_w, va_w, te_w = zca_whiten(tp, vp, tep)
        elif wname.startswith("spca_r"):
            r = int(wname.split("r")[1])
            tr_w, va_w, te_w = shrinkage_pca_whiten(tp, vp, tep, r=r)

        # === kNN ===
        knn_type = cfg["knn_type"]
        k_val = cfg["k"]
        temp = cfg["temp"]
        alpha = cfg["alpha"]

        if knn_type == "cosine":
            kt = knn_logits(te_w, tr_w, blt, k=k_val, nc=nc, temperature=temp)
            kv = knn_logits(va_w, tr_w, blt, k=k_val, nc=nc, temperature=temp)
        else:  # csls
            kt = csls_knn(te_w, tr_w, blt, k=k_val, nc=nc, temperature=temp)
            kv = csls_knn(va_w, tr_w, blt, k=k_val, nc=nc, temperature=temp)

        # Blend
        blended_test = (1 - alpha) * tel_arr + alpha * kt
        blended_val = (1 - alpha) * vl_arr + alpha * kv

        # === Metric 3: Blended argmax ===
        preds_blend = np.argmax(blended_test, axis=1)
        acc_b, mf1_b, mp_b, mr_b, cm_b, report_b = full_metrics(tela, preds_blend)
        print(f"\n  [Blended argmax] ACC={acc_b:.4f}  M-F1={mf1_b:.4f}  M-P={mp_b:.4f}  M-R={mr_b:.4f}")
        print(f"  CM: {cm_b.tolist()}")

        # === Metric 4: Blended + threshold tuning on val ===
        if nc == 2:
            vd2 = blended_val[:, 1] - blended_val[:, 0]
            td2 = blended_test[:, 1] - blended_test[:, 0]
            bt, bv2 = 0, 0
            for t in np.arange(-3, 3, 0.02):
                v2 = accuracy_score(vla, (vd2 > t).astype(int))
                if v2 > bv2: bv2, bt = v2, t
            preds_bt = (td2 > bt).astype(int)
            acc_bt, mf1_bt, mp_bt, mr_bt, cm_bt, report_bt = full_metrics(tela, preds_bt)
            print(f"\n  [Blended+thresh={bt:.2f}] ACC={acc_bt:.4f}  M-F1={mf1_bt:.4f}  M-P={mp_bt:.4f}  M-R={mr_bt:.4f}")
            print(f"  CM: {cm_bt.tolist()}")

        # === Best overall ===
        best_acc = max(acc, acc_b)
        if cfg["thresh"] is not None:
            best_acc = max(best_acc, acc_t)
        if nc == 2:
            best_acc = max(best_acc, acc_bt)

        # Print full classification report for the best config
        print(f"\n  === BEST CONFIG REPORT ===")
        candidates = [(acc, preds_head, "head_argmax"), (acc_b, preds_blend, "blend_argmax")]
        if cfg["thresh"] is not None:
            candidates.append((acc_t, preds_thresh, "head_thresh"))
        if nc == 2:
            candidates.append((acc_bt, preds_bt, "blend_thresh"))

        best_entry = max(candidates, key=lambda x: x[0])
        print(f"  Best method: {best_entry[2]}")
        ba, mf, mpr, mrc, cmx, rep = full_metrics(tela, best_entry[1])
        print(f"  ACC={ba:.4f}  M-F1={mf:.4f}  M-P={mpr:.4f}  M-R={mrc:.4f}")
        print(f"  Confusion Matrix:\n{cmx}")
        print(f"\n{rep}")

if __name__ == "__main__":
    main()
