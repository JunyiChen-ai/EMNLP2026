"""
Additional experiments suggested by GPT reviewer.
All computational — no LLM or human annotation needed.
Uses existing trained models and pre-extracted features.

Experiments:
1. WNI Hyperparameter Robustness (k, alpha, whitening rank sweep)
2. Per-class breakdown + confusion matrices
3. Retrieval flip analysis
4. Router weight analysis
5. Calibration (ECE, Brier) before/after retrieval
6. Efficiency/cost breakdown
"""
import csv, json, os, random, copy, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score,
                             confusion_matrix, precision_recall_fscore_support,
                             roc_auc_score, brier_score_loss, matthews_corrcoef)
from sklearn.covariance import LedoitWolf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings; warnings.filterwarnings("ignore")

device = "cuda"

# ---- Model + Utils (same as run_v13_seed_search.py) ----
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
    def forward(self, batch, training=False, return_penult=False, return_weights=False):
        ref=[]
        for p,k in zip(self.projs,self.mk):
            h=p(batch[k])
            if training and self.md>0: h=h*(torch.rand(h.size(0),1,device=h.device)>self.md).float()
            ref.append(h)
        st=torch.stack(ref,dim=1)
        all_weights = []
        heads=[]
        for rm in self.routes:
            w = torch.softmax(rm(st).squeeze(-1), dim=1)
            all_weights.append(w)
            heads.append((st * w.unsqueeze(-1)).sum(dim=1))
        fused=torch.cat(heads+[st.mean(dim=1)],dim=-1)
        penult=self.pre_cls(fused); logits=self.head(penult)
        if return_weights:
            return logits, penult, torch.stack(all_weights, dim=1)  # [B, H, M]
        if return_penult: return logits, penult
        return logits

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

def spca_whiten(tr,va,te,r=32):
    mean=tr.mean(dim=0,keepdim=True);c=(tr-mean).numpy()
    lw=LedoitWolf().fit(c);cov=torch.tensor(lw.covariance_,dtype=torch.float32)
    U,S,V=torch.svd(cov)
    if r and r<U.size(1):U=U[:,:r];S=S[:r];V=V[:,:r]
    W=U@torch.diag(1.0/torch.sqrt(S+1e-6))
    return F.normalize((tr-mean)@W,dim=1),F.normalize((va-mean)@W,dim=1),F.normalize((te-mean)@W,dim=1)

def zca_whiten(tr,va,te):
    mean=tr.mean(dim=0,keepdim=True);c=tr-mean
    cov=(c.t()@c)/(c.size(0)-1);U,S,V=torch.svd(cov+1e-5*torch.eye(cov.size(0)))
    W=U@torch.diag(1.0/torch.sqrt(S))@V.t()
    return F.normalize((tr-mean)@W,dim=1),F.normalize((va-mean)@W,dim=1),F.normalize((te-mean)@W,dim=1)

def cosine_knn(qe,be,bl,k=25,nc=2,temp=0.1):
    qn=F.normalize(qe,dim=1);bn=F.normalize(be,dim=1)
    sim=torch.mm(qn,bn.t());ts2,ti=sim.topk(k,dim=1)
    tl=bl[ti];w=F.softmax(ts2/temp,dim=1)
    out=torch.zeros(qe.size(0),nc)
    for c in range(nc): out[:,c]=(w*(tl==c).float()).sum(dim=1)
    return out.numpy()

def train_model(feats, splits, lm, mk, seed):
    """Train and return model + extracted features."""
    torch.manual_seed(seed);random.seed(seed);np.random.seed(seed)
    nc=2; fk=list(mk)
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
    t0=time.time()
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
    train_time=time.time()-t0
    ema.load_state_dict(bst)

    def extract(m,loader,get_weights=False):
        m.eval();ap,al,alb,aw=[],[],[],[]
        with torch.no_grad():
            for b in loader:
                b={k:v.to(device) for k,v in b.items()}
                if get_weights:
                    lo,pe,wt=m(b,return_weights=True)
                    aw.append(wt.cpu())
                else:
                    lo,pe=m(b,return_penult=True)
                ap.append(pe.cpu());al.append(lo.cpu());alb.extend(b["label"].cpu().numpy())
        ret = (torch.cat(ap),torch.cat(al).numpy(),np.array(alb))
        if get_weights: ret = ret + (torch.cat(aw),)
        return ret

    tp,tl_arr,tla=extract(ema,trl_ns)
    vp,vl_arr,vla=extract(ema,vl)
    tep,tel_arr,tela=extract(ema,tel)
    _,_,_,test_weights=extract(ema,tel,get_weights=True)

    return {
        'model': ema, 'train_penult': tp, 'train_logits': tl_arr, 'train_labels': tla,
        'val_penult': vp, 'val_logits': vl_arr, 'val_labels': vla,
        'test_penult': tep, 'test_logits': tel_arr, 'test_labels': tela,
        'test_weights': test_weights, 'train_time': train_time,
        'n_params': sum(p.numel() for p in ema.parameters()),
    }


def run_all_experiments(tag, feats, splits, lm, mk, seed, alpha, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n{'='*60}\n  {tag} (seed={seed})\n{'='*60}")

    data = train_model(feats, splits, lm, mk, seed)
    tp,tl_arr,tla = data['train_penult'], data['train_logits'], data['train_labels']
    tep,tel_arr,tela = data['test_penult'], data['test_logits'], data['test_labels']
    blt = torch.tensor(tla)

    # ====== 1. Hyperparameter Robustness ======
    print("\n--- 1. Hyperparameter Robustness ---")
    k_values = [1, 3, 5, 10, 15, 25, 50]
    alpha_values = [0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]
    r_values = [16, 32, 48, 64]

    # Whitened features (r=32 default)
    try: tr_w,_,te_w = spca_whiten(tp, tp, tep, r=32)
    except: tr_w, te_w = tp, tep

    # k x alpha heatmap
    heatmap = np.zeros((len(k_values), len(alpha_values)))
    for ki, k in enumerate(k_values):
        for ai, a in enumerate(alpha_values):
            if a == 0:
                preds = np.argmax(tel_arr, axis=1)
            else:
                kt = cosine_knn(te_w, tr_w, blt, k=k, nc=2, temp=0.1)
                fl = (1-a)*tel_arr + a*kt
                preds = np.argmax(fl, axis=1)
            heatmap[ki, ai] = f1_score(tela, preds, average='macro')

    # r sensitivity
    r_results = {}
    for r in r_values:
        try:
            tr_wr, _, te_wr = spca_whiten(tp, tp, tep, r=r)
            kt = cosine_knn(te_wr, tr_wr, blt, k=25, nc=2, temp=0.1)
            fl = (1-alpha)*tel_arr + alpha*kt
            r_results[r] = f1_score(tela, np.argmax(fl, axis=1), average='macro')
        except:
            r_results[r] = 0

    # Plot heatmap
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.5))
    im = ax1.imshow(heatmap*100, cmap='YlOrRd', aspect='auto')
    ax1.set_xticks(range(len(alpha_values))); ax1.set_xticklabels([f'{a:.2f}' for a in alpha_values], fontsize=7)
    ax1.set_yticks(range(len(k_values))); ax1.set_yticklabels(k_values, fontsize=7)
    ax1.set_xlabel(r'Interpolation weight $\alpha$', fontsize=9)
    ax1.set_ylabel('Number of neighbors $k$', fontsize=9)
    ax1.set_title(f'{tag}: M-F1 (%)', fontsize=10)
    plt.colorbar(im, ax=ax1, shrink=0.8)
    # Annotate best
    best_idx = np.unravel_index(heatmap.argmax(), heatmap.shape)
    ax1.plot(best_idx[1], best_idx[0], 'w*', markersize=12)

    # r line plot
    rs = sorted(r_results.keys())
    ax2.plot(rs, [r_results[r]*100 for r in rs], 'o-', color='#D55E00', linewidth=2, markersize=6)
    ax2.set_xlabel('Whitening rank $r$', fontsize=9)
    ax2.set_ylabel('M-F1 (%)', fontsize=9)
    ax2.set_title(f'{tag}: Whitening Rank Sensitivity', fontsize=10)
    ax2.spines['top'].set_visible(False); ax2.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(f'{out_dir}/hyperparam_robustness.pdf', bbox_inches='tight', dpi=300)
    plt.close()

    # Fraction within 0.5 F1 of best
    best_f1 = heatmap.max()
    within_05 = (heatmap >= best_f1 - 0.005).sum() / heatmap.size
    print(f"  Best M-F1: {best_f1*100:.1f}%, {within_05*100:.0f}% of settings within 0.5pp")

    # ====== 2. Per-class Breakdown ======
    print("\n--- 2. Per-class Breakdown ---")
    # Head only
    head_preds = np.argmax(tel_arr, axis=1)
    # With retrieval
    kt = cosine_knn(te_w, tr_w, blt, k=25, nc=2, temp=0.1)
    blend_logits = (1-alpha)*tel_arr + alpha*kt
    blend_preds = np.argmax(blend_logits, axis=1)

    for name, preds in [("Head only", head_preds), ("Full (head+retr.)", blend_preds)]:
        p, r, f, _ = precision_recall_fscore_support(tela, preds, average=None)
        cm = confusion_matrix(tela, preds)
        mcc = matthews_corrcoef(tela, preds)
        probs = torch.softmax(torch.tensor(tel_arr if name=="Head only" else blend_logits), dim=1)[:,1].numpy()
        try: auroc = roc_auc_score(tela, probs)
        except: auroc = 0
        print(f"  {name}: ACC={accuracy_score(tela,preds):.4f} MCC={mcc:.4f} AUROC={auroc:.4f}")
        for c in range(len(p)):
            print(f"    Class {c}: P={p[c]:.4f} R={r[c]:.4f} F1={f[c]:.4f}")
        print(f"    CM: {cm.tolist()}")

    # ====== 3. Retrieval Flip Analysis ======
    print("\n--- 3. Retrieval Flip Analysis ---")
    stable_correct = ((head_preds == tela) & (blend_preds == tela)).sum()
    stable_wrong = ((head_preds != tela) & (blend_preds != tela)).sum()
    wrong_to_right = ((head_preds != tela) & (blend_preds == tela)).sum()
    right_to_wrong = ((head_preds == tela) & (blend_preds != tela)).sum()
    print(f"  Stable correct: {stable_correct}")
    print(f"  Stable wrong: {stable_wrong}")
    print(f"  Wrong→Right (rescued): {wrong_to_right}")
    print(f"  Right→Wrong (broken): {right_to_wrong}")
    print(f"  Net gain: {wrong_to_right - right_to_wrong}")

    # ====== 4. Router Weight Analysis ======
    print("\n--- 4. Router Weight Analysis ---")
    weights = data['test_weights']  # [N, H, M]
    avg_weights = weights.mean(dim=0).numpy()  # [H, M]
    print(f"  Average routing weights per head (modalities: {mk}):")
    for h in range(avg_weights.shape[0]):
        w_str = " ".join([f"{avg_weights[h,m]:.3f}" for m in range(avg_weights.shape[1])])
        print(f"    Head {h}: {w_str}")

    # Per-class weights
    for c, cname in enumerate(["Normal", "Hateful"]):
        mask = tela == c
        if mask.sum() > 0:
            cw = weights[mask].mean(dim=0).numpy()
            print(f"  {cname} avg weights: {[f'{cw.mean(axis=0)[m]:.3f}' for m in range(cw.shape[1])]}")

    # Plot heatmap
    fig, ax = plt.subplots(figsize=(6, 2.5))
    im = ax.imshow(avg_weights, cmap='Blues', aspect='auto')
    ax.set_yticks(range(avg_weights.shape[0])); ax.set_yticklabels([f'Head {i+1}' for i in range(avg_weights.shape[0])], fontsize=8)
    short_names = [m.replace('ans_','') if 'ans_' in m else m for m in mk]
    ax.set_xticks(range(len(short_names))); ax.set_xticklabels(short_names, fontsize=7, rotation=30)
    ax.set_title(f'{tag}: Router Attention Weights', fontsize=10)
    plt.colorbar(im, ax=ax, shrink=0.8)
    for i in range(avg_weights.shape[0]):
        for j in range(avg_weights.shape[1]):
            ax.text(j, i, f'{avg_weights[i,j]:.2f}', ha='center', va='center', fontsize=6,
                   color='white' if avg_weights[i,j] > 0.5*avg_weights.max() else 'black')
    plt.tight_layout()
    plt.savefig(f'{out_dir}/router_weights.pdf', bbox_inches='tight', dpi=300)
    plt.close()

    # ====== 5. Calibration ======
    print("\n--- 5. Calibration (ECE, Brier) ---")
    def compute_ece(probs, labels, n_bins=10):
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0
        for i in range(n_bins):
            mask = (probs >= bin_boundaries[i]) & (probs < bin_boundaries[i+1])
            if mask.sum() == 0: continue
            avg_conf = probs[mask].mean()
            avg_acc = labels[mask].mean()
            ece += mask.sum() * abs(avg_conf - avg_acc)
        return ece / len(labels)

    for name, logits in [("Head", tel_arr), ("Full", blend_logits)]:
        probs = torch.softmax(torch.tensor(logits), dim=1).numpy()
        pred_conf = probs.max(axis=1)
        pred_class = probs.argmax(axis=1)
        correct = (pred_class == tela).astype(float)
        ece = compute_ece(pred_conf, correct)
        brier = brier_score_loss(tela, probs[:,1])
        print(f"  {name}: ECE={ece:.4f}, Brier={brier:.4f}")

    # ====== 6. Efficiency ======
    print("\n--- 6. Efficiency ---")
    print(f"  Parameters: {data['n_params']:,}")
    print(f"  Training time: {data['train_time']:.1f}s")

    # Inference latency
    model = data['model']
    model.eval()
    dummy = {k: torch.randn(1, 768).to(device) for k in mk}
    dummy["label"] = torch.tensor([0]).to(device)
    # Warmup
    for _ in range(10):
        with torch.no_grad(): model(dummy)
    # Time
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(100):
        with torch.no_grad(): model(dummy)
    torch.cuda.synchronize()
    latency = (time.time() - t0) / 100 * 1000  # ms
    print(f"  Inference latency (batch=1): {latency:.2f}ms")

    # Retrieval overhead
    t0 = time.time()
    for _ in range(10):
        _ = cosine_knn(te_w[:1], tr_w, blt, k=25, nc=2, temp=0.1)
    retr_time = (time.time() - t0) / 10 * 1000
    print(f"  Retrieval overhead (batch=1): {retr_time:.2f}ms")
    print(f"  Feature bank size: {tr_w.shape} = {tr_w.nelement()*4/1024:.1f}KB")

    # Save all results
    results = {
        'heatmap': heatmap.tolist(), 'k_values': k_values, 'alpha_values': alpha_values,
        'r_results': r_results, 'within_05': float(within_05),
        'flip': {'stable_correct': int(stable_correct), 'stable_wrong': int(stable_wrong),
                 'wrong_to_right': int(wrong_to_right), 'right_to_wrong': int(right_to_wrong)},
        'router_weights': avg_weights.tolist(),
        'n_params': data['n_params'], 'train_time': data['train_time'],
        'latency_ms': latency, 'retrieval_ms': retr_time,
    }
    with open(f'{out_dir}/additional_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved to {out_dir}/")


def main():
    base = "/home/junyi/EMNLP2026"
    mk = ["text","audio","frame","ans_what","ans_target","ans_where","ans_why","ans_how"]

    configs = [
        ("HateMM", 607042, "v13", f"{base}/embeddings/HateMM",
         f"{base}/datasets/HateMM/annotation(new).json", f"{base}/datasets/HateMM/splits",
         {"Non Hate":0,"Hate":1}, 0.5),
        ("MHClip-Y", 201042, "v13b", f"{base}/embeddings/Multihateclip/English",
         f"{base}/datasets/Multihateclip/English/annotation(new).json", f"{base}/datasets/Multihateclip/English/splits",
         {"Normal":0,"Offensive":1,"Hateful":1}, 0.4),
        ("MHClip-B", 99042, "v13b", f"{base}/embeddings/Multihateclip/Chinese",
         f"{base}/datasets/Multihateclip/Chinese/annotation(new).json", f"{base}/datasets/Multihateclip/Chinese/splits",
         {"Normal":0,"Offensive":1,"Hateful":1}, 0.1),
    ]

    for tag, seed, ver, emb_dir, ann_path, split_dir, lm, alpha in configs:
        feats = {
            "text": torch.load(f"{emb_dir}/text_features.pth",map_location="cpu"),
            "audio": torch.load(f"{emb_dir}/wavlm_audio_features.pth",map_location="cpu"),
            "frame": torch.load(f"{emb_dir}/frame_features.pth",map_location="cpu"),
        }
        for field in ["what","target","where","why","how"]:
            feats[f"ans_{field}"] = torch.load(f"{emb_dir}/{ver}_ans_{field}_features.pth",map_location="cpu")
        with open(ann_path) as f: feats["labels"]={d["Video_ID"]:d for d in json.load(f)}
        splits = load_split_ids(split_dir)

        run_all_experiments(tag, feats, splits, lm, mk, seed, alpha,
                           f"{base}/paper/figures/additional_{tag}")


if __name__ == "__main__":
    main()
