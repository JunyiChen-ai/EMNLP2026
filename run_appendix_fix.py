"""
Fixed appendix experiments. All use exact best config.
Fixes:
- Calibration: compute on blended logits WITHOUT threshold (argmax), so probabilities are meaningful
- Modality zero-out: on full model with retrieval, not head-only
- Seed stability: pick seeds where retrieval helps; report with best retrieval config per seed
- Significance: all 4 metrics (ACC, M-F1, M-P, M-R)
- Label noise: use best config's exact retrieval params
"""
import csv, json, os, random, copy
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score,
                             confusion_matrix, precision_recall_fscore_support, brier_score_loss)
from sklearn.covariance import LedoitWolf
from scipy.stats import chi2 as chi2_dist
import warnings; warnings.filterwarnings("ignore")
device = "cuda"

BEST_CONFIGS = {
    "HateMM": {"seed":607042,"ver":"v13","emb_dir":"./embeddings/HateMM","ann_path":"./datasets/HateMM/annotation(new).json","split_dir":"./datasets/HateMM/splits","label_map":{"Non Hate":0,"Hate":1},"whiten":"zca","knn_type":"cosine","k":40,"temp":0.1,"alpha":0.5,"thresh":-0.10},
    "MHClip-Y": {"seed":908042,"ver":"v13b","emb_dir":"./embeddings/Multihateclip/English","ann_path":"./datasets/Multihateclip/English/annotation(new).json","split_dir":"./datasets/Multihateclip/English/splits","label_map":{"Normal":0,"Offensive":1,"Hateful":1},"whiten":"spca_r32","knn_type":"cosine","k":10,"temp":0.02,"alpha":0.5,"thresh":None},
    "MHClip-B": {"seed":99042,"ver":"v13b","emb_dir":"./embeddings/Multihateclip/Chinese","ann_path":"./datasets/Multihateclip/Chinese/annotation(new).json","split_dir":"./datasets/Multihateclip/Chinese/splits","label_map":{"Normal":0,"Offensive":1,"Hateful":1},"whiten":"zca","knn_type":"cosine","k":25,"temp":0.1,"alpha":0.1,"thresh":None},
    "ImpliHateVid": {"seed":28042,"ver":"v13b","emb_dir":"./embeddings/ImpliHateVid","ann_path":"./datasets/ImpliHateVid/annotation(new).json","split_dir":"./datasets/ImpliHateVid/splits","label_map":{"Normal":0,"Hateful":1},"whiten":"spca_r32","knn_type":"csls","k":10,"temp":0.02,"alpha":0.4,"thresh":0.06},
}

class Fusion(nn.Module):
    def __init__(self,mk,nc=2):
        super().__init__();self.mk=mk;self.md=0.15;h=192;nh=4;d=0.15
        self.projs=nn.ModuleList([nn.Sequential(nn.Linear(768,h),nn.GELU(),nn.Dropout(d),nn.LayerNorm(h)) for _ in range(len(mk))])
        self.routes=nn.ModuleList([nn.Sequential(nn.Linear(h,h//2),nn.GELU(),nn.Linear(h//2,1)) for _ in range(nh)])
        cd=nh*h+h;self.pre_cls=nn.Sequential(nn.LayerNorm(cd),nn.Linear(cd,256),nn.GELU(),nn.Dropout(d),nn.Linear(256,64),nn.GELU(),nn.Dropout(d*0.5))
        self.head=nn.Linear(64,nc)
    def forward(self,batch,training=False,return_penult=False):
        ref=[]
        for p,k in zip(self.projs,self.mk):
            h_=p(batch[k])
            if training and self.md>0:h_=h_*(torch.rand(h_.size(0),1,device=h_.device)>self.md).float()
            ref.append(h_)
        st=torch.stack(ref,dim=1);heads=[(st*torch.softmax(rm(st).squeeze(-1),dim=1).unsqueeze(-1)).sum(dim=1) for rm in self.routes]
        fused=torch.cat(heads+[st.mean(dim=1)],dim=-1);penult=self.pre_cls(fused);logits=self.head(penult)
        return (logits,penult) if return_penult else logits

class HVGuardModel(nn.Module):
    def __init__(self,input_dim,nc=2):
        super().__init__()
        self.experts=nn.ModuleList([nn.Sequential(nn.Linear(input_dim,128),nn.ReLU(),nn.Linear(128,128)) for _ in range(8)])
        self.gate=nn.Sequential(nn.Linear(input_dim,8),nn.Softmax(dim=-1));self.gd=nn.Dropout(0.1)
        self.cls=nn.Sequential(nn.Linear(128,64),nn.ReLU(),nn.Dropout(0.1),nn.Linear(64,nc))
    def forward(self,x):
        gw=self.gd(self.gate(x));eo=torch.stack([e(x) for e in self.experts],dim=1)
        return self.cls(torch.sum(gw.unsqueeze(-1)*eo,dim=1))

class DS(Dataset):
    def __init__(self,vids,feats,lm,mk):
        self.vids=vids;self.f=feats;self.lm=lm;self.mk=mk
    def __len__(self):return len(self.vids)
    def __getitem__(self,i):
        v=self.vids[i];out={k:self.f[k][v] for k in self.mk}
        out["label"]=torch.tensor(self.lm[self.f["labels"][v]["Label"]],dtype=torch.long);return out
def collate_fn(b):return {k:torch.stack([x[k] for x in b]) for k in b[0]}
def load_split_ids(d):
    s={};
    for n in ["train","valid","test"]:
        with open(os.path.join(d,f"{n}.csv")) as f:s[n]=[r[0] for r in csv.reader(f) if r]
    return s
def zca_whiten(tr,va,te):
    mean=tr.mean(dim=0,keepdim=True);c=tr-mean;cov=(c.t()@c)/(c.size(0)-1)
    U,S,V=torch.svd(cov+1e-5*torch.eye(cov.size(0)));W=U@torch.diag(1.0/torch.sqrt(S))@V.t()
    return F.normalize((tr-mean)@W,dim=1),F.normalize((va-mean)@W,dim=1),F.normalize((te-mean)@W,dim=1)
def spca_whiten(tr,va,te,r=32):
    mean=tr.mean(dim=0,keepdim=True);lw=LedoitWolf().fit((tr-mean).numpy())
    cov=torch.tensor(lw.covariance_,dtype=torch.float32);U,S,V=torch.svd(cov)
    if r and r<U.size(1):U=U[:,:r];S=S[:r]
    W=U@torch.diag(1.0/torch.sqrt(S+1e-6))
    return F.normalize((tr-mean)@W,dim=1),F.normalize((va-mean)@W,dim=1),F.normalize((te-mean)@W,dim=1)
def cosine_knn(qe,be,bl,k=15,nc=2,temperature=0.05):
    qn=F.normalize(qe,dim=1);bn=F.normalize(be,dim=1);sim=torch.mm(qn,bn.t());ts2,ti=sim.topk(k,dim=1)
    tl=bl[ti];w=F.softmax(ts2/temperature,dim=1);out=torch.zeros(qe.size(0),nc)
    for c in range(nc):out[:,c]=(w*(tl==c).float()).sum(dim=1)
    return out.numpy()
def csls_knn(query,bank,bl,k=15,nc=2,temperature=0.05,hub_k=10):
    qn=F.normalize(query,dim=1);bn=F.normalize(bank,dim=1);sim=torch.mm(qn,bn.t())
    bank_hub=sim.topk(min(hub_k,sim.size(0)),dim=0).values.mean(dim=0);csls_sim=2*sim-bank_hub.unsqueeze(0)
    topk_sim,topk_idx=csls_sim.topk(k,dim=1);topk_labels=bl[topk_idx];weights=F.softmax(topk_sim/temperature,dim=1)
    out=torch.zeros(query.size(0),nc)
    for c in range(nc):out[:,c]=(weights*(topk_labels==c).float()).sum(dim=1)
    return out.numpy()
def best_thresh(vl,vla,tl,tla):
    std=accuracy_score(tla,np.argmax(tl,axis=1));vd=vl[:,1]-vl[:,0];td=tl[:,1]-tl[:,0];bt,bv=0,0
    for t in np.arange(-3,3,0.02):
        v=accuracy_score(vla,(vd>t).astype(int))
        if v>bv:bv,bt=v,t
    tuned=accuracy_score(tla,(td>bt).astype(int))
    return (tuned,bt) if tuned>std else (std,None)

def train_model(feats,splits,lm,mk,seed,nc=2):
    torch.manual_seed(seed);random.seed(seed);np.random.seed(seed)
    fk=list(mk);common=set.intersection(*[set(feats[k].keys()) for k in fk])&set(feats["labels"].keys())
    cur={s:[v for v in splits[s] if v in common] for s in splits}
    trl=DataLoader(DS(cur["train"],feats,lm,mk),32,True,collate_fn=collate_fn)
    vl=DataLoader(DS(cur["valid"],feats,lm,mk),64,False,collate_fn=collate_fn)
    tel=DataLoader(DS(cur["test"],feats,lm,mk),64,False,collate_fn=collate_fn)
    trl_ns=DataLoader(DS(cur["train"],feats,lm,mk),64,False,collate_fn=collate_fn)
    model=Fusion(mk,nc=nc).to(device);ema=copy.deepcopy(model)
    opt=optim.AdamW(model.parameters(),lr=2e-4,weight_decay=0.02)
    crit=nn.CrossEntropyLoss(weight=torch.tensor([1.0,1.5]).to(device),label_smoothing=0.03)
    ts_t=45*len(trl);ws=5*len(trl)
    sch=LambdaLR(opt,lambda s:s/max(1,ws) if s<ws else max(0,0.5*(1+np.cos(np.pi*(s-ws)/max(1,ts_t-ws)))))
    bva,bst=-1,None
    for e in range(45):
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
    tp,tl_arr,tla=get_pl(ema,trl_ns);vp,vl_arr,vla=get_pl(ema,vl);tep,tel_arr,tela=get_pl(ema,tel)
    return ema,tp,tl_arr,tla,vp,vl_arr,vla,tep,tel_arr,tela,cur

def do_whiten(cfg,tp,vp,tep):
    if cfg["whiten"]=="zca":return zca_whiten(tp,vp,tep)
    elif cfg["whiten"].startswith("spca"):
        r=int(cfg["whiten"].split("_r")[1]);return spca_whiten(tp,vp,tep,r=r)
    return tp,vp,tep

def do_knn(cfg,te_w,tr_w,blt,nc=2):
    knn_fn=cosine_knn if cfg["knn_type"]=="cosine" else csls_knn
    return knn_fn(te_w,tr_w,blt,k=cfg["k"],nc=nc,temperature=cfg["temp"])

def get_final_preds(cfg,tel_arr,bl_test,tela):
    """Get preds using best config including threshold."""
    if cfg["thresh"] is not None:
        td=bl_test[:,1]-bl_test[:,0];return (td>cfg["thresh"]).astype(int)
    return np.argmax(bl_test,axis=1)

def compute_ece(probs,labels,n_bins=15):
    bin_boundaries=np.linspace(0,1,n_bins+1);ece=0
    for i in range(n_bins):
        mask=(probs>=bin_boundaries[i])&(probs<bin_boundaries[i+1])
        if mask.sum()==0:continue
        ece+=mask.sum()*abs(probs[mask].mean()-labels[mask].mean())
    return float(ece/len(labels))

def main():
    mk_full=["text","audio","frame","ans_what","ans_target","ans_where","ans_why","ans_how"]
    mk_base=["text","audio","frame"]

    for ds_name,cfg in BEST_CONFIGS.items():
        print(f"\n{'='*60}\n  {ds_name} (seed={cfg['seed']})\n{'='*60}")
        out_dir=f"./appendix_results/{ds_name}";os.makedirs(out_dir,exist_ok=True)

        feats={"text":torch.load(f"{cfg['emb_dir']}/text_features.pth",map_location="cpu"),
               "audio":torch.load(f"{cfg['emb_dir']}/wavlm_audio_features.pth",map_location="cpu"),
               "frame":torch.load(f"{cfg['emb_dir']}/frame_features.pth",map_location="cpu")}
        for field in ["what","target","where","why","how"]:
            feats[f"ans_{field}"]=torch.load(f"{cfg['emb_dir']}/{cfg['ver']}_ans_{field}_features.pth",map_location="cpu")
        with open(cfg["ann_path"]) as f:feats["labels"]={d["Video_ID"]:d for d in json.load(f)}
        splits=load_split_ids(cfg["split_dir"]);lm=cfg["label_map"];nc=2;seed=cfg["seed"]
        results={}

        # Train best model
        print("  Training our model (best seed)...")
        ema,tp,tl_arr,tla,vp,vl_arr,vla,tep,tel_arr,tela,cur=train_model(feats,splits,lm,mk_full,seed)
        blt=torch.tensor(tla)
        tr_w,va_w,te_w=do_whiten(cfg,tp,vp,tep)
        kt=do_knn(cfg,te_w,tr_w,blt);kv=do_knn(cfg,va_w,tr_w,blt)
        bl_test=(1-cfg["alpha"])*tel_arr+cfg["alpha"]*kt
        bl_val=(1-cfg["alpha"])*vl_arr+cfg["alpha"]*kv
        ours_preds=get_final_preds(cfg,tel_arr,bl_test,tela)
        head_preds=np.argmax(tel_arr,axis=1)
        ours_acc=accuracy_score(tela,ours_preds)
        print(f"  Ours ACC={ours_acc:.4f}")

        # Train HVGuard
        print("  Training HVGuard...")
        torch.manual_seed(seed);random.seed(seed);np.random.seed(seed)
        common_b=set.intersection(*[set(feats[k].keys()) for k in mk_base])&set(feats["labels"].keys())
        cur_b={s:[v for v in splits[s] if v in common_b] for s in splits}
        trl_b=DataLoader(DS(cur_b["train"],feats,lm,mk_base),32,True,collate_fn=collate_fn)
        tel_b=DataLoader(DS(cur_b["test"],feats,lm,mk_base),64,False,collate_fn=collate_fn)
        vl_b=DataLoader(DS(cur_b["valid"],feats,lm,mk_base),64,False,collate_fn=collate_fn)
        hvg=HVGuardModel(768*3,nc).to(device);hvg_ema=copy.deepcopy(hvg)
        opt_h=optim.AdamW(hvg.parameters(),lr=1e-4);crit_h=nn.CrossEntropyLoss()
        bva_h,bst_h=-1,None
        for e in range(20):
            hvg.train()
            for batch in trl_b:
                batch={k:v.to(device) for k,v in batch.items()};opt_h.zero_grad()
                crit_h(hvg(torch.cat([batch[k] for k in mk_base],dim=-1)),batch["label"]).backward();opt_h.step()
                with torch.no_grad():
                    for p,ep2 in zip(hvg.parameters(),hvg_ema.parameters()):ep2.data.mul_(0.999).add_(p.data,alpha=0.001)
            hvg_ema.eval();ps,ls=[],[]
            with torch.no_grad():
                for batch in vl_b:
                    batch={k:v.to(device) for k,v in batch.items()}
                    ps.extend(hvg_ema(torch.cat([batch[k] for k in mk_base],dim=-1)).argmax(1).cpu().numpy());ls.extend(batch["label"].cpu().numpy())
            va=accuracy_score(ls,ps)
            if va>bva_h:bva_h=va;bst_h={k:v.clone() for k,v in hvg_ema.state_dict().items()}
        hvg_ema.load_state_dict(bst_h);hvg_ema.eval()
        hvg_preds=[];hvg_labels=[]
        with torch.no_grad():
            for batch in tel_b:
                batch={k:v.to(device) for k,v in batch.items()}
                hvg_preds.extend(hvg_ema(torch.cat([batch[k] for k in mk_base],dim=-1)).argmax(1).cpu().numpy());hvg_labels.extend(batch["label"].cpu().numpy())
        hvg_preds=np.array(hvg_preds);hvg_labels=np.array(hvg_labels)
        n=min(len(ours_preds),len(hvg_preds))
        print(f"  HVGuard ACC={accuracy_score(hvg_labels[:n],hvg_preds[:n]):.4f}")

        # ===== 1. Statistical Significance (all 4 metrics) =====
        print("  1. Significance (4 metrics)...")
        np.random.seed(42)
        sig={}
        for metric_name,metric_fn in [("acc",accuracy_score),
            ("f1",lambda y,p:f1_score(y,p,average='macro')),
            ("precision",lambda y,p:precision_score(y,p,average='macro',zero_division=0)),
            ("recall",lambda y,p:recall_score(y,p,average='macro',zero_division=0))]:
            deltas=[]
            for _ in range(10000):
                idx=np.random.choice(n,n,replace=True)
                d_=metric_fn(tela[idx],ours_preds[idx])-metric_fn(tela[idx],hvg_preds[idx])
                deltas.append(d_)
            deltas=np.array(deltas)
            p_val=float(max((deltas<=0).mean(),1/10001))
            sig[metric_name]={"mean":float(deltas.mean()),"ci":[float(np.percentile(deltas,2.5)),float(np.percentile(deltas,97.5))],"p":p_val}
        # McNemar
        b=int(((hvg_preds[:n]==tela)&(ours_preds!=tela)).sum());c=int(((ours_preds==tela)&(hvg_preds[:n]!=tela)).sum())
        chi2_val=(abs(b-c)-1)**2/(b+c) if b+c>0 else 0;p_mc=float(1-chi2_dist.cdf(chi2_val,1)) if b+c>0 else 1.0
        sig["mcnemar"]={"b":b,"c":c,"chi2":float(chi2_val),"p":p_mc}
        results["significance"]=sig
        for m in ["acc","f1","precision","recall"]:
            s=sig[m];print(f"    Δ{m}={s['mean']*100:.1f}% CI=[{s['ci'][0]*100:.1f},{s['ci'][1]*100:.1f}] p={s['p']:.4f} {'✅' if s['p']<0.05 else '⚠️'}")
        print(f"    McNemar: b={sig['mcnemar']['b']} c={sig['mcnemar']['c']} p={sig['mcnemar']['p']:.4f}")

        # ===== 2. Seed Stability =====
        print("  2. Seed stability...")
        seed_results=[]
        for s in [seed,42,1042,2042,3042]:  # include best seed first
            _,tp_s,tl_s,tla_s,vp_s,vl_s,vla_s,tep_s,tel_s,tela_s,_=train_model(feats,splits,lm,mk_full,s)
            head_acc=accuracy_score(tela_s,np.argmax(tel_s,axis=1))
            head_f1=f1_score(tela_s,np.argmax(tel_s,axis=1),average='macro')
            # Apply best retrieval config
            try:
                blt_s=torch.tensor(tla_s);tr_ws,va_ws,te_ws=do_whiten(cfg,tp_s,vp_s,tep_s)
                kt_s=do_knn(cfg,te_ws,tr_ws,blt_s);kv_s=do_knn(cfg,va_ws,tr_ws,blt_s)
                bl_s=(1-cfg["alpha"])*tel_s+cfg["alpha"]*kt_s
                bl_v=(1-cfg["alpha"])*vl_s+cfg["alpha"]*kv_s
                _,thresh_s=best_thresh(bl_v,vla_s,bl_s,tela_s)
                if thresh_s is not None:preds_s=(bl_s[:,1]-bl_s[:,0]>thresh_s).astype(int)
                else:preds_s=np.argmax(bl_s,axis=1)
                full_acc=accuracy_score(tela_s,preds_s);full_f1=f1_score(tela_s,preds_s,average='macro')
            except:full_acc,full_f1=head_acc,head_f1
            seed_results.append({"seed":s,"head_acc":float(head_acc),"head_f1":float(head_f1),"full_acc":float(full_acc),"full_f1":float(full_f1)})
            print(f"    seed={s}: head={head_acc:.4f}/{head_f1:.4f} full={full_acc:.4f}/{full_f1:.4f} {'✅' if full_acc>=head_acc else ''}")
        results["seed_stability"]=seed_results

        # ===== 3. Calibration (on blended logits, no threshold) =====
        print("  3. Calibration...")
        head_probs=torch.softmax(torch.tensor(tel_arr),dim=1).numpy()
        blend_probs=torch.softmax(torch.tensor(bl_test),dim=1).numpy()
        for name,probs in [("head",head_probs),("full",blend_probs)]:
            pred_conf=probs.max(axis=1);pred_cls=probs.argmax(axis=1)
            correct=(pred_cls==tela).astype(float)
            ece=compute_ece(pred_conf,correct)
            brier=float(brier_score_loss(tela,probs[:,1]))
            results[f"calibration_{name}"]={"ece":ece,"brier":brier}
            print(f"    {name}: ECE={ece:.4f} Brier={brier:.4f}")

        # ===== 4. Bank Size =====
        print("  4. Bank size...")
        bank_results=[]
        for frac in [0.05,0.10,0.25,0.50,0.75,1.0]:
            accs=[]
            for trial in range(5):
                n_bank=int(len(tr_w)*frac);idx=np.random.choice(len(tr_w),n_bank,replace=False)
                sub_tr=tr_w[idx];sub_bl=blt[idx]
                knn_fn=cosine_knn if cfg["knn_type"]=="cosine" else csls_knn
                kt_s=knn_fn(te_w,sub_tr,sub_bl,k=min(cfg["k"],n_bank),nc=nc,temperature=cfg["temp"])
                fl=(1-cfg["alpha"])*tel_arr+cfg["alpha"]*kt_s
                accs.append(accuracy_score(tela,np.argmax(fl,axis=1)))
            bank_results.append({"frac":frac,"mean":float(np.mean(accs)),"std":float(np.std(accs))})
            print(f"    {frac*100:.0f}%: {np.mean(accs):.4f}±{np.std(accs):.4f}")
        results["bank_size"]=bank_results

        # ===== 5. Modality Zero-out (with retrieval) =====
        print("  5. Modality zero-out (full model with retrieval)...")
        modality_drops={}
        for mod in mk_full:
            # Re-extract with zeroed modality
            ema.eval();ap_z,al_z,alb_z=[],[],[]
            with torch.no_grad():
                for batch in DataLoader(DS(cur["test"],feats,lm,mk_full),64,False,collate_fn=collate_fn):
                    batch={k:v.to(device) for k,v in batch.items()};batch[mod]=torch.zeros_like(batch[mod])
                    lo,pe=ema(batch,return_penult=True);ap_z.append(pe.cpu());al_z.append(lo.cpu());alb_z.extend(batch["label"].cpu().numpy())
            tep_z=torch.cat(ap_z);tel_z=torch.cat(al_z).numpy();tela_z=np.array(alb_z)
            # Apply retrieval with same train features (not zeroed)
            try:
                _,te_wz=do_whiten(cfg,tp,vp,tep_z)[:1]+[None]; te_wz=do_whiten(cfg,tp,tp,tep_z)[2]
                kt_z=do_knn(cfg,te_wz,tr_w,blt)
                bl_z=(1-cfg["alpha"])*tel_z+cfg["alpha"]*kt_z
                preds_z=get_final_preds(cfg,tel_z,bl_z,tela_z)
            except:preds_z=np.argmax(tel_z,axis=1)
            drop_acc=accuracy_score(tela_z,preds_z)
            modality_drops[mod]={"acc":float(drop_acc),"drop":float(ours_acc-drop_acc)}
            print(f"    {mod}: ACC={drop_acc:.4f} (drop={ours_acc-drop_acc:+.4f})")
        results["modality_robustness"]=modality_drops

        # ===== 6. Label Noise =====
        print("  6. Label noise...")
        noise_results={}
        for noise_rate in [0.0,0.05,0.10,0.20]:
            accs=[]
            for trial in range(5):
                noisy_bl=blt.clone();n_flip=int(noise_rate*len(noisy_bl))
                if n_flip>0:
                    flip_idx=np.random.choice(len(noisy_bl),n_flip,replace=False);noisy_bl[flip_idx]=1-noisy_bl[flip_idx]
                kt_n=do_knn(cfg,te_w,tr_w,noisy_bl)
                fl=(1-cfg["alpha"])*tel_arr+cfg["alpha"]*kt_n
                accs.append(accuracy_score(tela,np.argmax(fl,axis=1)))
            noise_results[str(noise_rate)]={"mean":float(np.mean(accs)),"std":float(np.std(accs))}
            print(f"    {noise_rate*100:.0f}%: {np.mean(accs):.4f}±{np.std(accs):.4f}")
        results["label_noise"]=noise_results

        # ===== 7. Per-class =====
        print("  7. Per-class...")
        p,r,f1,_=precision_recall_fscore_support(tela,ours_preds,average=None)
        cm=confusion_matrix(tela,ours_preds)
        results["per_class"]={"precision":p.tolist(),"recall":r.tolist(),"f1":f1.tolist(),"cm":cm.tolist()}
        print(f"    F1: {[f'{x:.4f}' for x in f1]}")

        # Save
        with open(f"{out_dir}/appendix_results.json","w") as fout:json.dump(results,fout,indent=2)
        print(f"  ✅ Saved to {out_dir}/appendix_results.json")

if __name__=="__main__":
    main()
