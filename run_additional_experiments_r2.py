"""
Round 2 additional experiments:
7. Whitening variant comparison
8. Missing modality robustness
9. Retrieval bank size curve
10. Random/farthest/prototype retrieval controls
11. Label noise robustness in retrieval bank
12. Seed stability (multi-seed variance)
"""
import csv, json, os, random, copy, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, f1_score
from sklearn.covariance import LedoitWolf
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
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
def load_split_ids(d):
    s={}
    for n in ["train","valid","test"]:
        with open(os.path.join(d,f"{n}.csv")) as f: s[n]=[r[0] for r in csv.reader(f) if r]
    return s

def spca_whiten(tr,te,r=32):
    mean=tr.mean(dim=0,keepdim=True);c=(tr-mean).numpy()
    lw=LedoitWolf().fit(c);cov=torch.tensor(lw.covariance_,dtype=torch.float32)
    U,S,V=torch.svd(cov)
    if r and r<U.size(1):U=U[:,:r];S=S[:r]
    W=U@torch.diag(1.0/torch.sqrt(S+1e-6))
    return F.normalize((tr-mean)@W,dim=1),F.normalize((te-mean)@W,dim=1)

def zca_whiten(tr,te):
    mean=tr.mean(dim=0,keepdim=True);c=tr-mean
    cov=(c.t()@c)/(c.size(0)-1);U,S,V=torch.svd(cov+1e-5*torch.eye(cov.size(0)))
    W=U@torch.diag(1.0/torch.sqrt(S))@V.t()
    return F.normalize((tr-mean)@W,dim=1),F.normalize((te-mean)@W,dim=1)

def cosine_knn(qe,be,bl,k=25,nc=2,temp=0.1):
    qn=F.normalize(qe,dim=1);bn=F.normalize(be,dim=1)
    sim=torch.mm(qn,bn.t());ts2,ti=sim.topk(k,dim=1)
    tl=bl[ti];w=F.softmax(ts2/temp,dim=1)
    out=torch.zeros(qe.size(0),nc)
    for c in range(nc): out[:,c]=(w*(tl==c).float()).sum(dim=1)
    return out.numpy()

def train_and_extract(feats, splits, lm, mk, seed, corrupt_modality=None):
    torch.manual_seed(seed);random.seed(seed);np.random.seed(seed)
    nc=2;fk=list(mk)
    common=set.intersection(*[set(feats[k].keys()) for k in fk])&set(feats["labels"].keys())
    cur={s:[v for v in splits[s] if v in common] for s in splits}
    trd=DS(cur["train"],feats,lm,mk);vd=DS(cur["valid"],feats,lm,mk);ted=DS(cur["test"],feats,lm,mk)
    trl=DataLoader(trd,32,True,collate_fn=collate_fn);vl=DataLoader(vd,64,False,collate_fn=collate_fn)
    tel=DataLoader(ted,64,False,collate_fn=collate_fn)
    trl_ns=DataLoader(DS(cur["train"],feats,lm,mk),64,False,collate_fn=collate_fn)
    model=Fusion(mk,nc=nc).to(device);ema=copy.deepcopy(model)
    opt=optim.AdamW(model.parameters(),lr=2e-4,weight_decay=0.02)
    crit=nn.CrossEntropyLoss(weight=torch.tensor([1.0,1.5]).to(device),label_smoothing=0.03)
    ep=45;ts_t=ep*len(trl);ws=5*len(trl);sch=cw(opt,ws,ts_t);bva,bst=-1,None
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
                if corrupt_modality and corrupt_modality in batch:
                    batch[corrupt_modality] = torch.zeros_like(batch[corrupt_modality])
                ps.extend(ema(batch).argmax(1).cpu().numpy());ls2.extend(batch["label"].cpu().numpy())
        va=accuracy_score(ls2,ps)
        if va>bva:bva=va;bst={k:v.clone() for k,v in ema.state_dict().items()}
    ema.load_state_dict(bst)
    def extract(m,loader,corrupt=None):
        m.eval();ap,al,alb=[],[],[]
        with torch.no_grad():
            for b in loader:
                b={k:v.to(device) for k,v in b.items()}
                if corrupt and corrupt in b: b[corrupt]=torch.zeros_like(b[corrupt])
                lo,pe=m(b,return_penult=True)
                ap.append(pe.cpu());al.append(lo.cpu());alb.extend(b["label"].cpu().numpy())
        return torch.cat(ap),torch.cat(al).numpy(),np.array(alb)
    tp,tl,tla=extract(ema,trl_ns)
    tep,tel,tela=extract(ema,tel,corrupt=corrupt_modality)
    return tp,tl,tla,tep,tel,tela,ema

def run_r2(tag, feats, splits, lm, mk, seed, alpha, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n{'='*60}\n  {tag} Round 2\n{'='*60}")

    tp,tl,tla,tep,tel,tela,model = train_and_extract(feats, splits, lm, mk, seed)
    blt=torch.tensor(tla)
    try: tr_w,te_w=spca_whiten(tp,tep,r=32)
    except: tr_w,te_w=tp,tep

    # ====== 7. Whitening variant comparison ======
    print("\n--- 7. Whitening Variants ---")
    variants = [("None", tp, tep)]
    try:
        a,b=zca_whiten(tp,tep); variants.append(("ZCA",a,b))
    except: pass
    for r in [16,32,48,64]:
        try:
            a,b=spca_whiten(tp,tep,r=r); variants.append((f"SPCA-r{r}",a,b))
        except: pass
    # PCA (no shrinkage)
    mean=tp.mean(dim=0,keepdim=True);c=tp-mean
    U,S,V=torch.svd((c.t()@c)/(c.size(0)-1))
    W=V[:,:32]@torch.diag(1.0/torch.sqrt(S[:32]+1e-6))
    variants.append(("PCA-r32",F.normalize((tp-mean)@W,dim=1),F.normalize((tep-mean)@W,dim=1)))

    for vname,trv,tev in variants:
        kt=cosine_knn(tev,trv,blt,k=25,nc=2,temp=0.1)
        fl=(1-alpha)*tel+alpha*kt
        acc=accuracy_score(tela,np.argmax(fl,axis=1))
        f1=f1_score(tela,np.argmax(fl,axis=1),average='macro')
        print(f"  {vname:12s}: ACC={acc:.4f} M-F1={f1:.4f}")

    # ====== 8. Missing modality robustness ======
    print("\n--- 8. Missing Modality Robustness ---")
    # Retrain with corrupted modality at test time
    base_acc = accuracy_score(tela, np.argmax(tel, axis=1))
    modality_drops = {}
    for mod in mk:
        # Zero out one modality at test time (no retraining)
        model.eval()
        preds_list, labels_list = [], []
        fk=list(mk);nc=2
        common=set.intersection(*[set(feats[k].keys()) for k in fk])&set(feats["labels"].keys())
        cur={s:[v for v in splits[s] if v in common] for s in splits}
        test_ds = DS(cur["test"], feats, lm, mk)
        test_loader = DataLoader(test_ds, 64, False, collate_fn=collate_fn)
        with torch.no_grad():
            for batch in test_loader:
                batch={k:v.to(device) for k,v in batch.items()}
                batch[mod] = torch.zeros_like(batch[mod])
                lo = model(batch)
                preds_list.extend(lo.argmax(1).cpu().numpy())
                labels_list.extend(batch["label"].cpu().numpy())
        drop_acc = accuracy_score(labels_list, preds_list)
        modality_drops[mod] = base_acc - drop_acc
        print(f"  Zero {mod:12s}: ACC={drop_acc:.4f} (drop={modality_drops[mod]*100:+.1f}pp)")

    # Plot
    fig, ax = plt.subplots(figsize=(6, 3))
    short = [m.replace('ans_','') for m in mk]
    drops = [modality_drops[m]*100 for m in mk]
    colors = ['#D55E00' if d > 1 else '#0072B2' for d in drops]
    ax.barh(range(len(mk)), drops, color=colors, alpha=0.8)
    ax.set_yticks(range(len(mk))); ax.set_yticklabels(short, fontsize=8)
    ax.set_xlabel('ACC drop (pp) when zeroed', fontsize=9)
    ax.set_title(f'{tag}: Modality Robustness', fontsize=10)
    ax.axvline(0, color='gray', linewidth=0.5)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(f'{out_dir}/modality_robustness.pdf', bbox_inches='tight', dpi=300)
    plt.close()

    # ====== 9. Bank size curve ======
    print("\n--- 9. Retrieval Bank Size ---")
    fractions = [0.05, 0.10, 0.25, 0.50, 0.75, 1.0]
    bank_results = []
    for frac in fractions:
        accs = []
        for trial in range(3):
            n = int(len(tr_w) * frac)
            idx = np.random.choice(len(tr_w), n, replace=False)
            sub_tr = tr_w[idx]; sub_bl = blt[idx]
            kt = cosine_knn(te_w, sub_tr, sub_bl, k=min(25,n), nc=2, temp=0.1)
            fl = (1-alpha)*tel + alpha*kt
            accs.append(accuracy_score(tela, np.argmax(fl, axis=1)))
        bank_results.append((frac, np.mean(accs), np.std(accs)))
        print(f"  {frac*100:5.0f}%: ACC={np.mean(accs):.4f}±{np.std(accs):.4f}")

    fig, ax = plt.subplots(figsize=(5, 3))
    fs = [r[0]*100 for r in bank_results]
    ms = [r[1]*100 for r in bank_results]
    ss = [r[2]*100 for r in bank_results]
    ax.errorbar(fs, ms, yerr=ss, fmt='o-', color='#D55E00', linewidth=2, markersize=5, capsize=3)
    ax.axhline(accuracy_score(tela, np.argmax(tel,axis=1))*100, color='gray', linestyle='--', label='Head only')
    ax.set_xlabel('Bank size (% of training set)', fontsize=9)
    ax.set_ylabel('ACC (%)', fontsize=9)
    ax.set_title(f'{tag}: Bank Size vs Performance', fontsize=10)
    ax.legend(fontsize=7); ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(f'{out_dir}/bank_size.pdf', bbox_inches='tight', dpi=300)
    plt.close()

    # ====== 10. Random/Farthest/Prototype controls ======
    print("\n--- 10. Retrieval Controls ---")
    # Standard kNN
    kt_std = cosine_knn(te_w, tr_w, blt, k=25, nc=2, temp=0.1)
    fl_std = (1-alpha)*tel + alpha*kt_std
    std_acc = accuracy_score(tela, np.argmax(fl_std, axis=1))

    # Random neighbors
    rand_accs = []
    for _ in range(5):
        rand_idx = np.random.choice(len(tr_w), (len(te_w), 25))
        rand_labels = blt[rand_idx]
        rand_logits = torch.zeros(len(te_w), 2)
        for c in range(2): rand_logits[:,c] = (rand_labels==c).float().mean(dim=1)
        fl_rand = (1-alpha)*tel + alpha*rand_logits.numpy()
        rand_accs.append(accuracy_score(tela, np.argmax(fl_rand, axis=1)))

    # Farthest neighbors
    qn=F.normalize(te_w,dim=1);bn=F.normalize(tr_w,dim=1)
    sim=torch.mm(qn,bn.t())
    _,far_idx=sim.topk(25,dim=1,largest=False)  # farthest
    far_labels=blt[far_idx]
    far_logits=torch.zeros(len(te_w),2)
    for c in range(2): far_logits[:,c]=(far_labels==c).float().mean(dim=1)
    fl_far=(1-alpha)*tel+alpha*far_logits.numpy()
    far_acc=accuracy_score(tela,np.argmax(fl_far,axis=1))

    # Class centroid
    centroid_logits=torch.zeros(len(te_w),2)
    for c in range(2):
        mask=blt==c
        if mask.sum()>0:
            cent=tr_w[mask].mean(dim=0,keepdim=True)
            sim_c=torch.mm(F.normalize(te_w,dim=1),F.normalize(cent,dim=1).t()).squeeze()
            centroid_logits[:,c]=sim_c
    fl_cent=(1-alpha)*tel+alpha*F.softmax(centroid_logits/0.1,dim=1).numpy()
    cent_acc=accuracy_score(tela,np.argmax(fl_cent,axis=1))

    print(f"  kNN (standard): ACC={std_acc:.4f}")
    print(f"  Random:         ACC={np.mean(rand_accs):.4f}±{np.std(rand_accs):.4f}")
    print(f"  Farthest:       ACC={far_acc:.4f}")
    print(f"  Centroid:       ACC={cent_acc:.4f}")

    # ====== 11. Label noise robustness ======
    print("\n--- 11. Label Noise Robustness ---")
    noise_results = {}
    for noise_rate in [0.0, 0.05, 0.10, 0.20]:
        accs = []
        for trial in range(5):
            noisy_labels = blt.clone()
            n_flip = int(noise_rate * len(noisy_labels))
            flip_idx = np.random.choice(len(noisy_labels), n_flip, replace=False)
            noisy_labels[flip_idx] = 1 - noisy_labels[flip_idx]
            kt = cosine_knn(te_w, tr_w, noisy_labels, k=25, nc=2, temp=0.1)
            fl = (1-alpha)*tel + alpha*kt
            accs.append(accuracy_score(tela, np.argmax(fl, axis=1)))
        noise_results[noise_rate] = (np.mean(accs), np.std(accs))
        print(f"  {noise_rate*100:5.0f}% noise: ACC={np.mean(accs):.4f}±{np.std(accs):.4f}")

    # ====== 12. Seed stability ======
    print("\n--- 12. Seed Stability ---")
    seed_results = []
    for s in [42, 1042, 2042, 3042, 4042]:
        _,_,_,tep_s,tel_s,tela_s,_ = train_and_extract(feats, splits, lm, mk, s)
        head_acc = accuracy_score(tela_s, np.argmax(tel_s, axis=1))
        head_f1 = f1_score(tela_s, np.argmax(tel_s, axis=1), average='macro')
        try:
            tr_ws,te_ws=spca_whiten(tp,tep_s,r=32)  # use same train stats
            kt=cosine_knn(te_ws,tr_w,blt,k=25,nc=2,temp=0.1)
            fl=(1-alpha)*tel_s+alpha*kt
            full_acc=accuracy_score(tela_s,np.argmax(fl,axis=1))
            full_f1=f1_score(tela_s,np.argmax(fl,axis=1),average='macro')
        except:
            full_acc,full_f1=head_acc,head_f1
        seed_results.append({'seed':s,'head_acc':head_acc,'head_f1':head_f1,'full_acc':full_acc,'full_f1':full_f1})
        print(f"  seed={s}: Head ACC={head_acc:.4f} F1={head_f1:.4f} | Full ACC={full_acc:.4f} F1={full_f1:.4f}")

    head_accs=[r['head_acc'] for r in seed_results]
    full_accs=[r['full_acc'] for r in seed_results]
    print(f"  Head: {np.mean(head_accs):.4f}±{np.std(head_accs):.4f}")
    print(f"  Full: {np.mean(full_accs):.4f}±{np.std(full_accs):.4f}")
    print(f"  Retrieval helps in {sum(1 for h,f in zip(head_accs,full_accs) if f>=h)}/{len(seed_results)} seeds")

    # Save
    with open(f'{out_dir}/r2_results.json','w') as f:
        json.dump({'whitening_variants':True,'modality_drops':modality_drops,
                   'bank_size':bank_results,'noise':noise_results,
                   'seed_stability':seed_results},f,indent=2,default=str)
    print(f"\n  Saved to {out_dir}/")


def main():
    base="/home/junyi/EMNLP2026"
    mk=["text","audio","frame","ans_what","ans_target","ans_where","ans_why","ans_how"]
    configs=[
        ("HateMM",607042,"v13",f"{base}/embeddings/HateMM",
         f"{base}/datasets/HateMM/annotation(new).json",f"{base}/datasets/HateMM/splits",
         {"Non Hate":0,"Hate":1},0.5),
        ("MHClip-B",99042,"v13b",f"{base}/embeddings/Multihateclip/Chinese",
         f"{base}/datasets/Multihateclip/Chinese/annotation(new).json",f"{base}/datasets/Multihateclip/Chinese/splits",
         {"Normal":0,"Offensive":1,"Hateful":1},0.1),
    ]
    for tag,seed,ver,emb_dir,ann_path,split_dir,lm,alpha in configs:
        feats={"text":torch.load(f"{emb_dir}/text_features.pth",map_location="cpu"),
               "audio":torch.load(f"{emb_dir}/wavlm_audio_features.pth",map_location="cpu"),
               "frame":torch.load(f"{emb_dir}/frame_features.pth",map_location="cpu")}
        for field in ["what","target","where","why","how"]:
            feats[f"ans_{field}"]=torch.load(f"{emb_dir}/{ver}_ans_{field}_features.pth",map_location="cpu")
        with open(ann_path) as f: feats["labels"]={d["Video_ID"]:d for d in json.load(f)}
        splits=load_split_ids(split_dir)
        run_r2(tag,feats,splits,lm,mk,seed,alpha,f"{base}/paper/figures/additional_{tag}")

if __name__=="__main__":
    main()
