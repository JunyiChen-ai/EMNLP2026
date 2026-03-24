"""
Main training/evaluation for v13 (Perception-to-Cognition).

6 modalities:
- text (768d): BERT encode title+transcript
- audio (768d): WavLM encode audio
- frame (768d): ViT encode frames
- perception (768d): BERT encode step1+step2 (scene + evidence)
- cognition (768d): BERT encode step3+step4 (target/intent + harm/norm)
- answer (768d): BERT encode full answer

struct: 1d (is_hateful from LLM)

Same fusion architecture as before: Multi-Head Gated Routing + modality dropout.
Same training: AdamW, cosine warmup, EMA, label smoothing.
Same inference: optional whitening + kNN retrieval.
"""
import argparse, csv, json, os, random, copy
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.covariance import LedoitWolf
from torch.utils.data import DataLoader, Dataset
import warnings; warnings.filterwarnings("ignore")

device = "cuda"

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
    def __init__(self, mk, hidden=192, nh=4, sd=1, nc=2, drop=0.15, md=0.15):
        super().__init__()
        nm = len(mk); self.mk = mk; self.md = md
        self.projs = nn.ModuleList([nn.Sequential(
            nn.Linear(768, hidden), nn.GELU(), nn.Dropout(drop), nn.LayerNorm(hidden)
        ) for _ in range(nm)])
        self.se = nn.Sequential(nn.Linear(sd, 16), nn.GELU(), nn.LayerNorm(16))
        self.routes = nn.ModuleList([nn.Sequential(
            nn.Linear(hidden, hidden//2), nn.GELU(), nn.Linear(hidden//2, 1)
        ) for _ in range(nh)])
        cd = nh * hidden + hidden + 16
        self.pre_cls = nn.Sequential(
            nn.LayerNorm(cd), nn.Linear(cd, 256), nn.GELU(), nn.Dropout(drop),
            nn.Linear(256, 64), nn.GELU(), nn.Dropout(drop * 0.5)
        )
        self.head = nn.Linear(64, nc)

    def forward(self, batch, training=False, return_penult=False):
        ref = []
        for i, (p, k) in enumerate(zip(self.projs, self.mk)):
            h = p(batch[k])
            if training and self.md > 0:
                h = h * (torch.rand(h.size(0), 1, device=h.device) > self.md).float()
            ref.append(h)
        st = torch.stack(ref, dim=1)
        heads = [((st * torch.softmax(rm(st).squeeze(-1), dim=1).unsqueeze(-1)).sum(dim=1))
                 for rm in self.routes]
        fused = torch.cat(heads + [st.mean(dim=1), self.se(batch["struct"])], dim=-1)
        penult = self.pre_cls(fused)
        logits = self.head(penult)
        if return_penult:
            return logits, penult
        return logits

def cw(opt, ws, ts):
    def f(s):
        if s < ws: return s / max(1, ws)
        return max(0, 0.5 * (1 + np.cos(np.pi * (s - ws) / max(1, ts - ws))))
    return LambdaLR(opt, f)

def load_split_ids(d):
    s = {}
    for n in ["train", "valid", "test"]:
        with open(os.path.join(d, f"{n}.csv")) as f:
            s[n] = [r[0] for r in csv.reader(f) if r]
    return s

def get_pl(model, loader):
    model.eval(); ap, al, alb = [], [], []
    with torch.no_grad():
        for b in loader:
            b = {k: v.to(device) for k, v in b.items()}
            lo, pe = model(b, return_penult=True)
            ap.append(pe.cpu()); al.append(lo.cpu()); alb.extend(b["label"].cpu().numpy())
    return torch.cat(ap), torch.cat(al).numpy(), np.array(alb)

def spca_whiten(tr, va, te, r=32):
    mean = tr.mean(dim=0, keepdim=True)
    c = (tr - mean).numpy()
    lw = LedoitWolf().fit(c)
    cov = torch.tensor(lw.covariance_, dtype=torch.float32)
    U, S, V = torch.svd(cov)
    if r and r < U.size(1): U = U[:, :r]; S = S[:r]; V = V[:, :r]
    W = U @ torch.diag(1.0 / torch.sqrt(S + 1e-6))
    return (F.normalize((tr - mean) @ W, dim=1),
            F.normalize((va - mean) @ W, dim=1),
            F.normalize((te - mean) @ W, dim=1))

def knn_logits(qe, be, bl, k=10, nc=2, temperature=0.05):
    qn = F.normalize(qe, dim=1); bn = F.normalize(be, dim=1)
    sim = torch.mm(qn, bn.t()); ts2, ti = sim.topk(k, dim=1)
    tl = bl[ti]; w = F.softmax(ts2 / temperature, dim=1)
    out = torch.zeros(qe.size(0), nc)
    for c in range(nc): out[:, c] = (w * (tl == c).float()).sum(dim=1)
    return out.numpy()

def train_and_eval(feats, cur, lm, mk, sd, nc, seed, class_weight=None,
                   whiten_r=32, knn_k=10, knn_temp=0.05, knn_alpha=0.0):
    torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
    trd = DS(cur["train"], feats, lm, mk)
    vd = DS(cur["valid"], feats, lm, mk)
    ted = DS(cur["test"], feats, lm, mk)
    trl = DataLoader(trd, 32, True, collate_fn=collate_fn)
    vl = DataLoader(vd, 64, False, collate_fn=collate_fn)
    tel = DataLoader(ted, 64, False, collate_fn=collate_fn)
    trl_ns = DataLoader(DS(cur["train"], feats, lm, mk), 64, False, collate_fn=collate_fn)

    model = Fusion(mk, hidden=192, sd=sd, nc=nc, drop=0.15, md=0.15).to(device)
    ema = copy.deepcopy(model)
    opt = optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.02)
    if class_weight:
        crit = nn.CrossEntropyLoss(weight=torch.tensor(class_weight, dtype=torch.float).to(device),
                                   label_smoothing=0.03)
    else:
        crit = nn.CrossEntropyLoss(label_smoothing=0.03)
    ep = 45; ts_t = ep * len(trl); ws = 5 * len(trl); sch = cw(opt, ws, ts_t)
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
    tp, tl_arr, tla = get_pl(ema, trl_ns)
    vp, vl_arr, vla = get_pl(ema, vl)
    tep, tel_arr, tela = get_pl(ema, tel)
    blt = torch.tensor(tla)

    # Head only
    preds_head = np.argmax(tel_arr, axis=1)
    head_acc = accuracy_score(tela, preds_head)
    head_f1 = f1_score(tela, preds_head, average='macro')
    head_p = precision_score(tela, preds_head, average='macro')
    head_r = recall_score(tela, preds_head, average='macro')

    # With kNN
    if knn_alpha > 0:
        try:
            tr_w, va_w, te_w = spca_whiten(tp, vp, tep, r=whiten_r)
        except:
            tr_w, va_w, te_w = tp, vp, tep
        kt = knn_logits(te_w, tr_w, blt, k=knn_k, nc=nc, temperature=knn_temp)
        fl = (1 - knn_alpha) * tel_arr + knn_alpha * kt
        preds_knn = np.argmax(fl, axis=1)
        knn_acc = accuracy_score(tela, preds_knn)
        knn_f1 = f1_score(tela, preds_knn, average='macro')
        knn_p = precision_score(tela, preds_knn, average='macro')
        knn_r = recall_score(tela, preds_knn, average='macro')
    else:
        knn_acc, knn_f1, knn_p, knn_r = head_acc, head_f1, head_p, head_r

    cm = confusion_matrix(tela, preds_knn if knn_alpha > 0 else preds_head)
    return head_acc, head_f1, head_p, head_r, knn_acc, knn_f1, knn_p, knn_r, cm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", default="HateMM", choices=["HateMM", "Multihateclip"])
    parser.add_argument("--language", default="English")
    parser.add_argument("--num_seeds", type=int, default=5)
    parser.add_argument("--knn_alpha", type=float, default=0.2)
    args = parser.parse_args()

    if args.dataset_name == "HateMM":
        emb_dir = "./embeddings/HateMM"
        ann_path = "./datasets/HateMM/annotation(new).json"
        split_dir = "./datasets/HateMM/splits"
        lm = {"Non Hate": 0, "Hate": 1}
    else:
        emb_dir = f"./embeddings/Multihateclip/{args.language}"
        ann_path = f"./datasets/Multihateclip/{args.language}/annotation(new).json"
        split_dir = f"./datasets/Multihateclip/{args.language}/splits"
        lm = {"Normal": 0, "Offensive": 1, "Hateful": 1}

    nc = 2
    mk = ["text", "audio", "frame", "perception", "cognition", "answer"]

    feats = {
        "text": torch.load(f"{emb_dir}/text_features.pth", map_location="cpu"),
        "audio": torch.load(f"{emb_dir}/wavlm_audio_features.pth", map_location="cpu"),
        "frame": torch.load(f"{emb_dir}/frame_features.pth", map_location="cpu"),
        "perception": torch.load(f"{emb_dir}/v13_perception_features.pth", map_location="cpu"),
        "cognition": torch.load(f"{emb_dir}/v13_cognition_features.pth", map_location="cpu"),
        "answer": torch.load(f"{emb_dir}/v13_answer_features.pth", map_location="cpu"),
        "struct": torch.load(f"{emb_dir}/v13_struct_features.pth", map_location="cpu"),
    }
    with open(ann_path) as f:
        feats["labels"] = {d["Video_ID"]: d for d in json.load(f)}

    splits = load_split_ids(split_dir)
    fk = mk + ["struct"]
    common = set.intersection(*[set(feats[k].keys()) for k in fk]) & set(feats["labels"].keys())
    cur = {s: [v for v in splits[s] if v in common] for s in splits}
    sd = feats["struct"][list(feats["struct"].keys())[0]].shape[0]

    print(f"\n{'='*70}")
    print(f"  v13 P2C — {args.dataset_name} {args.language}")
    print(f"  Train: {len(cur['train'])}, Val: {len(cur['valid'])}, Test: {len(cur['test'])}")
    print(f"  Modalities: {mk}, struct_dim={sd}")
    print(f"{'='*70}")

    seeds = [42, 1042, 2042, 3042, 4042][:args.num_seeds]
    all_results = []

    for seed in seeds:
        ha, hf, hp, hr, ka, kf, kp, kr, cm = train_and_eval(
            feats, cur, lm, mk, sd, nc, seed,
            class_weight=[1.0, 1.5],
            whiten_r=32, knn_k=10, knn_temp=0.05, knn_alpha=args.knn_alpha
        )
        print(f"  seed={seed}: Head ACC={ha:.4f} M-F1={hf:.4f} | +kNN ACC={ka:.4f} M-F1={kf:.4f} M-P={kp:.4f} M-R={kr:.4f}")
        all_results.append((ha, hf, hp, hr, ka, kf, kp, kr))

    arr = np.array(all_results)
    m = arr.mean(axis=0); s = arr.std(axis=0)
    print(f"\n  Mean (5 seeds):")
    print(f"    Head:  ACC={m[0]:.4f}±{s[0]:.4f}  M-F1={m[1]:.4f}±{s[1]:.4f}")
    print(f"    +kNN:  ACC={m[4]:.4f}±{s[4]:.4f}  M-F1={m[5]:.4f}±{s[5]:.4f}  M-P={m[6]:.4f}±{s[6]:.4f}  M-R={m[7]:.4f}±{s[7]:.4f}")
    print(f"    Max:   Head ACC={arr[:,0].max():.4f}  +kNN ACC={arr[:,4].max():.4f}")

if __name__ == "__main__":
    main()
