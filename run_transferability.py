"""
Cross-dataset transferability experiments.
6 transfer directions: each of 3 datasets as source, test on the other 2 targets.
Methods: Ours (with retrieval from source), HVGuard, ImpliHateVid.
Metrics: ACC, M-F1, M-P, M-R.
Generates radar figure.
"""
import argparse, csv, json, os, random, copy
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.covariance import LedoitWolf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import warnings; warnings.filterwarnings("ignore")

device = "cuda"

# ---- Data ----
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

# ---- Our Fusion Model ----
class Fusion(nn.Module):
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

def cw(opt,ws,ts):
    def f(s):
        if s<ws: return s/max(1,ws)
        return max(0,0.5*(1+np.cos(np.pi*(s-ws)/max(1,ts-ws))))
    return LambdaLR(opt,f)

# ---- HVGuard Model ----
class HVGuardModel(nn.Module):
    def __init__(self, input_dim, nc=2):
        super().__init__()
        experts = nn.ModuleList([nn.Sequential(nn.Linear(input_dim,128),nn.ReLU(),nn.Linear(128,128)) for _ in range(8)])
        gate = nn.Sequential(nn.Linear(input_dim,8),nn.Softmax(dim=-1))
        self.experts=experts; self.gate=gate; self.gate_drop=nn.Dropout(0.1)
        self.cls=nn.Sequential(nn.Linear(128,64),nn.ReLU(),nn.Dropout(0.1),nn.Linear(64,nc))
    def forward(self, x):
        gw=self.gate_drop(self.gate(x))
        eo=torch.stack([e(x) for e in self.experts],dim=1)
        return self.cls(torch.sum(gw.unsqueeze(-1)*eo,dim=1))

# ---- ImpliHateVid Model ----
class ImpliHateVidModel(nn.Module):
    def __init__(self, input_dim=768, nc=2):
        super().__init__()
        self.img_enc=nn.Sequential(nn.Linear(input_dim,1024),nn.ReLU(),nn.Dropout(0.2),nn.Linear(1024,512),nn.ReLU(),nn.Dropout(0.2),nn.Linear(512,256),nn.ReLU(),nn.Dropout(0.2))
        self.txt_enc=nn.Sequential(nn.Linear(input_dim,1024),nn.ReLU(),nn.Dropout(0.2),nn.Linear(1024,512),nn.ReLU(),nn.Dropout(0.2),nn.Linear(512,256),nn.ReLU(),nn.Dropout(0.2))
        self.aud_enc=nn.Sequential(nn.Linear(input_dim,1024),nn.ReLU(),nn.Dropout(0.2),nn.Linear(1024,512),nn.ReLU(),nn.Dropout(0.2),nn.Linear(512,256),nn.ReLU(),nn.Dropout(0.2))
        self.cross_encs=nn.ModuleList([nn.Sequential(nn.Linear(256,256),nn.ReLU(),nn.Dropout(0.2),nn.Linear(256,128),nn.ReLU(),nn.Dropout(0.2)) for _ in range(6)])
        self.cls=nn.Sequential(nn.Linear(6*128,1024),nn.ReLU(),nn.Dropout(0.3),nn.Linear(1024,512),nn.ReLU(),nn.Dropout(0.3),nn.Linear(512,256),nn.ReLU(),nn.Dropout(0.3),nn.Linear(256,nc))
    def forward(self, img, txt, aud):
        i,t,a=self.img_enc(img),self.txt_enc(txt),self.aud_enc(aud)
        cross=[self.cross_encs[0](i),self.cross_encs[1](i),self.cross_encs[2](t),self.cross_encs[3](t),self.cross_encs[4](a),self.cross_encs[5](a)]
        return self.cls(torch.cat(cross,dim=-1))

# ---- Whitening + kNN for our method ----
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

# ---- Train + Transfer Eval ----
def get_pl(model, loader, model_type, mk):
    model.eval(); ap,al,alb=[],[],[]
    with torch.no_grad():
        for b in loader:
            b={k:v.to(device) for k,v in b.items()}
            if model_type == 'ours':
                lo,pe=model(b,return_penult=True)
            elif model_type == 'hvguard':
                combined=torch.cat([b[k] for k in mk],dim=-1)
                lo=model(combined); pe=lo  # no penult for hvguard
            elif model_type == 'impli':
                lo=model(b['frame'],b['text'],b['audio']); pe=lo
            ap.append(pe.cpu());al.append(lo.cpu());alb.extend(b["label"].cpu().numpy())
    return torch.cat(ap),torch.cat(al).numpy(),np.array(alb)

def train_and_transfer(source_feats, source_splits, target_feats, target_splits,
                       lm_source, lm_target, mk, model_type, seed, alpha=0.3):
    torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
    nc = 2

    # Build datasets
    fk = list(mk)
    src_common = set.intersection(*[set(source_feats[k].keys()) for k in fk]) & set(source_feats["labels"].keys())
    tgt_common = set.intersection(*[set(target_feats[k].keys()) for k in fk]) & set(target_feats["labels"].keys())
    src_cur = {s:[v for v in source_splits[s] if v in src_common] for s in source_splits}
    tgt_test = [v for v in target_splits["test"] if v in tgt_common]

    src_trd = DS(src_cur["train"], source_feats, lm_source, mk)
    src_vd = DS(src_cur["valid"], source_feats, lm_source, mk)
    tgt_ted = DS(tgt_test, target_feats, lm_target, mk)
    trl = DataLoader(src_trd, 32, True, collate_fn=collate_fn)
    vl = DataLoader(src_vd, 64, False, collate_fn=collate_fn)
    tel = DataLoader(tgt_ted, 64, False, collate_fn=collate_fn)

    # Build model
    if model_type == 'ours':
        model = Fusion(mk, nc=nc).to(device)
    elif model_type == 'hvguard':
        input_dim = sum(source_feats[k][list(source_feats[k].keys())[0]].shape[-1] for k in mk)
        model = HVGuardModel(input_dim, nc).to(device)
    elif model_type == 'impli':
        model = ImpliHateVidModel(768, nc).to(device)

    ema = copy.deepcopy(model)
    opt = optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.02)
    crit = nn.CrossEntropyLoss(weight=torch.tensor([1.0,1.5]).to(device), label_smoothing=0.03)
    ep=45; ts_t=ep*len(trl); ws=5*len(trl); sch=cw(opt,ws,ts_t)
    bva, bst = -1, None

    for e in range(ep):
        model.train()
        for batch in trl:
            batch={k:v.to(device) for k,v in batch.items()}; opt.zero_grad()
            if model_type == 'ours':
                lo = model(batch, training=True)
            elif model_type == 'hvguard':
                combined = torch.cat([batch[k] for k in mk], dim=-1)
                lo = model(combined)
            elif model_type == 'impli':
                lo = model(batch['frame'], batch['text'], batch['audio'])
            crit(lo, batch["label"]).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); sch.step()
            with torch.no_grad():
                for p,ep2 in zip(model.parameters(),ema.parameters()):ep2.data.mul_(0.999).add_(p.data,alpha=0.001)
        ema.eval(); ps,ls2=[],[]
        with torch.no_grad():
            for batch in vl:
                batch={k:v.to(device) for k,v in batch.items()}
                if model_type == 'ours': lo=ema(batch)
                elif model_type == 'hvguard': lo=ema(torch.cat([batch[k] for k in mk],dim=-1))
                elif model_type == 'impli': lo=ema(batch['frame'],batch['text'],batch['audio'])
                ps.extend(lo.argmax(1).cpu().numpy()); ls2.extend(batch["label"].cpu().numpy())
        va=accuracy_score(ls2,ps)
        if va>bva: bva=va; bst={k:v.clone() for k,v in ema.state_dict().items()}

    ema.load_state_dict(bst)

    # Evaluate on target test
    if model_type == 'ours':
        # With retrieval from source
        src_trl_ns = DataLoader(DS(src_cur["train"],source_feats,lm_source,mk),64,False,collate_fn=collate_fn)
        tp,tl_arr,tla = get_pl(ema, src_trl_ns, 'ours', mk)
        _,tel_arr,tela = get_pl(ema, tel, 'ours', mk)
        # Also get target penultimate
        tep_list = []
        ema.eval()
        with torch.no_grad():
            for b in tel:
                b={k:v.to(device) for k,v in b.items()}
                _,pe=ema(b,return_penult=True)
                tep_list.append(pe.cpu())
        tep = torch.cat(tep_list)
        blt = torch.tensor(tla)
        try:
            tr_w,_,te_w = zca_whiten(tp, tp, tep)  # whiten using source stats
        except:
            tr_w, te_w = tp, tep
        kt = cosine_knn(te_w, tr_w, blt, k=25, nc=nc, temp=0.1)
        fl = (1-alpha)*tel_arr + alpha*kt
        preds = np.argmax(fl, axis=1)
    else:
        ema.eval(); preds_list, labels_list = [], []
        with torch.no_grad():
            for batch in tel:
                batch={k:v.to(device) for k,v in batch.items()}
                if model_type == 'hvguard': lo=ema(torch.cat([batch[k] for k in mk],dim=-1))
                elif model_type == 'impli': lo=ema(batch['frame'],batch['text'],batch['audio'])
                preds_list.extend(lo.argmax(1).cpu().numpy())
                labels_list.extend(batch["label"].cpu().numpy())
        preds = np.array(preds_list)
        tela = np.array(labels_list)

    return {
        "acc": float(accuracy_score(tela, preds)),
        "f1": float(f1_score(tela, preds, average='macro')),
        "p": float(precision_score(tela, preds, average='macro', zero_division=0)),
        "r": float(recall_score(tela, preds, average='macro', zero_division=0)),
    }

def load_dataset_feats(ds_name):
    base = "/home/junyi/EMNLP2026"
    if ds_name == "HateMM":
        emb_dir = f"{base}/embeddings/HateMM"
        ann_path = f"{base}/datasets/HateMM/annotation(new).json"
        split_dir = f"{base}/datasets/HateMM/splits"
        lm = {"Non Hate":0, "Hate":1}
        ver = "v13"
    elif ds_name == "MHClip-Y":
        emb_dir = f"{base}/embeddings/Multihateclip/English"
        ann_path = f"{base}/datasets/Multihateclip/English/annotation(new).json"
        split_dir = f"{base}/datasets/Multihateclip/English/splits"
        lm = {"Normal":0, "Offensive":1, "Hateful":1}
        ver = "v13b"
    elif ds_name == "MHClip-B":
        emb_dir = f"{base}/embeddings/Multihateclip/Chinese"
        ann_path = f"{base}/datasets/Multihateclip/Chinese/annotation(new).json"
        split_dir = f"{base}/datasets/Multihateclip/Chinese/splits"
        lm = {"Normal":0, "Offensive":1, "Hateful":1}
        ver = "v13b"

    feats = {
        "text": torch.load(f"{emb_dir}/text_features.pth",map_location="cpu"),
        "audio": torch.load(f"{emb_dir}/wavlm_audio_features.pth",map_location="cpu"),
        "frame": torch.load(f"{emb_dir}/frame_features.pth",map_location="cpu"),
    }
    for field in ["what","target","where","why","how"]:
        feats[f"ans_{field}"] = torch.load(f"{emb_dir}/{ver}_ans_{field}_features.pth",map_location="cpu")
    with open(ann_path) as f:
        feats["labels"] = {d["Video_ID"]:d for d in json.load(f)}
    splits = load_split_ids(split_dir)
    return feats, splits, lm

def main():
    datasets = ["HateMM", "MHClip-Y", "MHClip-B"]
    mk_ours = ["text","audio","frame","ans_what","ans_target","ans_where","ans_why","ans_how"]
    mk_base = ["text","audio","frame"]
    seed = 42

    # Load all datasets
    all_data = {}
    for ds in datasets:
        all_data[ds] = load_dataset_feats(ds)

    results = {}
    for src in datasets:
        for tgt in datasets:
            if src == tgt: continue
            key = f"{src}->{tgt}"
            print(f"\n{'='*50}")
            print(f"  Transfer: {key}")

            src_feats, src_splits, src_lm = all_data[src]
            tgt_feats, tgt_splits, tgt_lm = all_data[tgt]

            results[key] = {}

            # Ours
            r = train_and_transfer(src_feats, src_splits, tgt_feats, tgt_splits,
                                   src_lm, tgt_lm, mk_ours, 'ours', seed)
            results[key]['Ours'] = r
            print(f"  Ours:      ACC={r['acc']:.4f} F1={r['f1']:.4f} P={r['p']:.4f} R={r['r']:.4f}")

            # HVGuard
            r = train_and_transfer(src_feats, src_splits, tgt_feats, tgt_splits,
                                   src_lm, tgt_lm, mk_base, 'hvguard', seed)
            results[key]['HVGuard'] = r
            print(f"  HVGuard:   ACC={r['acc']:.4f} F1={r['f1']:.4f} P={r['p']:.4f} R={r['r']:.4f}")

            # ImpliHateVid
            r = train_and_transfer(src_feats, src_splits, tgt_feats, tgt_splits,
                                   src_lm, tgt_lm, mk_base, 'impli', seed)
            results[key]['ImpliHateVid'] = r
            print(f"  ImpliHate: ACC={r['acc']:.4f} F1={r['f1']:.4f} P={r['p']:.4f} R={r['r']:.4f}")

    # Save results
    os.makedirs("transfer_results", exist_ok=True)
    with open("transfer_results/all_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Generate radar figure
    fig, axes = plt.subplots(2, 3, figsize=(14, 9), subplot_kw=dict(polar=True))
    metrics = ['ACC', 'M-F1', 'M-P', 'M-R']
    methods = ['Ours', 'HVGuard', 'ImpliHateVid']
    colors = ['#E74C3C', '#4A90D9', '#2ECC71']
    angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]

    transfer_keys = list(results.keys())
    for idx, key in enumerate(transfer_keys):
        row, col = idx // 3, idx % 3
        ax = axes[row][col]
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_thetagrids(np.degrees(angles[:-1]), metrics, fontsize=8)

        for mi, method in enumerate(methods):
            if method in results[key]:
                r = results[key][method]
                values = [r['acc'], r['f1'], r['p'], r['r']]
                values += values[:1]
                ax.plot(angles, values, 'o-', linewidth=1.5, color=colors[mi], label=method, markersize=3)
                ax.fill(angles, values, alpha=0.08, color=colors[mi])

        ax.set_ylim(0, 1.0)
        ax.set_title(key, fontsize=10, fontweight='bold', pad=15)
        if idx == 0:
            ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.15), fontsize=7)

    plt.tight_layout()
    plt.savefig("paper/figures/transferability_radar.pdf", bbox_inches='tight', dpi=300)
    plt.savefig("paper/figures/transferability_radar.png", bbox_inches='tight', dpi=300)
    print("\nSaved paper/figures/transferability_radar.pdf")
    print("\nAll transfer results:")
    for key, methods_r in results.items():
        print(f"  {key}:")
        for m, r in methods_r.items():
            print(f"    {m:15s}: ACC={r['acc']:.4f} F1={r['f1']:.4f} P={r['p']:.4f} R={r['r']:.4f}")

if __name__ == "__main__":
    main()
