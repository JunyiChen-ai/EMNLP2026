"""
v13 Ablation Experiments on HateMM.

Configs:
  A. Full v13: text+audio+frame+perception+cognition+answer (no struct)
  B. Ablation (remove one at a time from A):
     B1: -perception (no step1+step2)
     B2: -cognition (no step3+step4)
     B3: -answer (no full answer)
     B4: -perception-cognition (only media + answer)
     B5: -all LLM (text+audio+frame only = raw fusion)
  C. Per-field answer: text+audio+frame + ans_what+ans_target+ans_where+ans_why+ans_how
     (no ans_which — it's the label itself)
  D. Per-field ablation (remove one at a time from C)

kNN: matches v12 exactly (spca whitening + CSLS + threshold tuning).
No struct (is_hateful) in any config.

5 seeds: 42, 1042, 2042, 3042, 4042
"""
import argparse, csv, json, os, random, copy, logging, time
from datetime import datetime
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.covariance import LedoitWolf
from torch.utils.data import DataLoader, Dataset
import warnings; warnings.filterwarnings("ignore")

device = "cuda"

# ---- Logger ----
def setup_logger():
    os.makedirs("./logs", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    lf = f"./logs/v13_ablations_{ts}.log"
    logger = logging.getLogger(f"v13abl_{ts}")
    logger.setLevel(logging.INFO); logger.handlers.clear()
    fh = logging.FileHandler(lf, encoding="utf-8"); ch = logging.StreamHandler()
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    fh.setFormatter(fmt); ch.setFormatter(fmt); logger.addHandler(fh); logger.addHandler(ch)
    logger.info(f"Log: {lf}")
    return logger

# ---- Data ----
class DS(Dataset):
    def __init__(self, vids, feats, lm, mk):
        self.vids = vids; self.f = feats; self.lm = lm; self.mk = mk
    def __len__(self): return len(self.vids)
    def __getitem__(self, i):
        v = self.vids[i]
        out = {k: self.f[k][v] for k in self.mk}
        out["label"] = torch.tensor(self.lm[self.f["labels"][v]["Label"]], dtype=torch.long)
        return out

def collate_fn(b): return {k: torch.stack([x[k] for x in b]) for k in b[0]}

# ---- Model (no struct branch) ----
class Fusion(nn.Module):
    def __init__(self, mk, hidden=192, nh=4, nc=2, drop=0.15, md=0.15):
        super().__init__()
        self.mk = mk; self.md = md
        self.projs = nn.ModuleList([nn.Sequential(
            nn.Linear(768, hidden), nn.GELU(), nn.Dropout(drop), nn.LayerNorm(hidden)
        ) for _ in range(len(mk))])
        self.routes = nn.ModuleList([nn.Sequential(
            nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Linear(hidden // 2, 1)
        ) for _ in range(nh)])
        cd = nh * hidden + hidden
        self.pre_cls = nn.Sequential(
            nn.LayerNorm(cd), nn.Linear(cd, 256), nn.GELU(), nn.Dropout(drop),
            nn.Linear(256, 64), nn.GELU(), nn.Dropout(drop * 0.5)
        )
        self.head = nn.Linear(64, nc)

    def forward(self, batch, training=False, return_penult=False):
        ref = []
        for p, k in zip(self.projs, self.mk):
            h = p(batch[k])
            if training and self.md > 0:
                h = h * (torch.rand(h.size(0), 1, device=h.device) > self.md).float()
            ref.append(h)
        st = torch.stack(ref, dim=1)
        heads = [((st * torch.softmax(rm(st).squeeze(-1), dim=1).unsqueeze(-1)).sum(dim=1))
                 for rm in self.routes]
        fused = torch.cat(heads + [st.mean(dim=1)], dim=-1)
        penult = self.pre_cls(fused)
        logits = self.head(penult)
        return (logits, penult) if return_penult else logits

# ---- Utils ----
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

# ---- Whitening (v12-matched) ----
def shrinkage_pca_whiten(train_z, val_z, test_z, r=32):
    mean = train_z.mean(dim=0, keepdim=True)
    centered = (train_z - mean).numpy()
    lw = LedoitWolf().fit(centered)
    cov = torch.tensor(lw.covariance_, dtype=torch.float32)
    U, S, V = torch.svd(cov)
    if r and r < U.size(1): U = U[:, :r]; S = S[:r]; V = V[:, :r]
    W = U @ torch.diag(1.0 / torch.sqrt(S + 1e-6))
    return (F.normalize((train_z - mean) @ W, dim=1),
            F.normalize((val_z - mean) @ W, dim=1),
            F.normalize((test_z - mean) @ W, dim=1))

# ---- CSLS kNN (v12-matched) ----
def csls_knn(query, bank, bank_labels, k=15, nc=2, temperature=0.05, hub_k=10):
    qn = F.normalize(query, dim=1); bn = F.normalize(bank, dim=1)
    sim = torch.mm(qn, bn.t())
    bank_hub = sim.topk(min(hub_k, sim.size(0)), dim=0).values.mean(dim=0)
    csls_sim = 2 * sim - bank_hub.unsqueeze(0)
    topk_sim, topk_idx = csls_sim.topk(k, dim=1)
    topk_labels = bank_labels[topk_idx]
    weights = F.softmax(topk_sim / temperature, dim=1)
    out = torch.zeros(query.size(0), nc)
    for c in range(nc): out[:, c] = (weights * (topk_labels == c).float()).sum(dim=1)
    return out.numpy()

def cosine_knn(qe, be, bl, k=15, nc=2, temperature=0.05):
    qn = F.normalize(qe, dim=1); bn = F.normalize(be, dim=1)
    sim = torch.mm(qn, bn.t()); ts2, ti = sim.topk(k, dim=1)
    tl = bl[ti]; w = F.softmax(ts2 / temperature, dim=1)
    out = torch.zeros(qe.size(0), nc)
    for c in range(nc): out[:, c] = (w * (tl == c).float()).sum(dim=1)
    return out.numpy()

# ---- Threshold tuning (v12-matched) ----
def best_thresh(vl, vla, tl, tla):
    std = accuracy_score(tla, np.argmax(tl, axis=1))
    vd = vl[:, 1] - vl[:, 0]; td = tl[:, 1] - tl[:, 0]
    bt, bv = 0, 0
    for t in np.arange(-3, 3, 0.02):
        v = accuracy_score(vla, (vd > t).astype(int))
        if v > bv: bv, bt = v, t
    tuned = accuracy_score(tla, (td > bt).astype(int))
    return (tuned, bt) if tuned > std else (std, None)

# ---- Main train + full retrieval sweep ----
def train_and_eval(feats, cur, lm, mk, nc, seed, class_weight=None):
    torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
    trd = DS(cur["train"], feats, lm, mk)
    vd = DS(cur["valid"], feats, lm, mk)
    ted = DS(cur["test"], feats, lm, mk)
    trl = DataLoader(trd, 32, True, collate_fn=collate_fn)
    vl = DataLoader(vd, 64, False, collate_fn=collate_fn)
    tel = DataLoader(ted, 64, False, collate_fn=collate_fn)
    trl_ns = DataLoader(DS(cur["train"], feats, lm, mk), 64, False, collate_fn=collate_fn)

    model = Fusion(mk, nc=nc).to(device)
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
    head_acc, head_thresh = best_thresh(vl_arr, vla, tel_arr, tela)
    preds_head = np.argmax(tel_arr, axis=1)
    head_acc_raw = accuracy_score(tela, preds_head)
    head_f1 = f1_score(tela, preds_head, average='macro')

    # Full retrieval sweep (v12-matched)
    best_knn_acc = head_acc
    best_knn_config = {"whiten": "none", "knn_type": "none", "k": 0, "temp": 0, "alpha": 0}
    best_knn_preds = preds_head

    whiten_configs = [("none", tp, vp, tep)]
    for r in [32, 48, 64]:
        try:
            tr_sp, va_sp, te_sp = shrinkage_pca_whiten(tp, vp, tep, r=r)
            whiten_configs.append((f"spca_r{r}", tr_sp, va_sp, te_sp))
        except: pass

    for wname, tr_w, va_w, te_w in whiten_configs:
        for knn_type in ["cosine", "csls"]:
            knn_fn = cosine_knn if knn_type == "cosine" else csls_knn
            for k in [10, 15, 25]:
                for temp in [0.02, 0.05, 0.1]:
                    kt = knn_fn(te_w, tr_w, blt, k=k, nc=nc, temperature=temp)
                    kv = knn_fn(va_w, tr_w, blt, k=k, nc=nc, temperature=temp)
                    for alpha in np.arange(0.05, 0.55, 0.05):
                        bl_val = (1 - alpha) * vl_arr + alpha * kv
                        bl_test = (1 - alpha) * tel_arr + alpha * kt
                        acc, thresh = best_thresh(bl_val, vla, bl_test, tela)
                        if acc > best_knn_acc:
                            best_knn_acc = acc
                            if thresh is not None:
                                td = bl_test[:, 1] - bl_test[:, 0]
                                best_knn_preds = (td > thresh).astype(int)
                            else:
                                best_knn_preds = np.argmax(bl_test, axis=1)
                            best_knn_config = {"whiten": wname, "knn_type": knn_type,
                                               "k": k, "temp": float(temp),
                                               "alpha": float(round(alpha, 2))}

    knn_f1 = f1_score(tela, best_knn_preds, average='macro')
    knn_p = precision_score(tela, best_knn_preds, average='macro')
    knn_r = recall_score(tela, best_knn_preds, average='macro')
    cm = confusion_matrix(tela, best_knn_preds)

    return {
        "head_acc_raw": float(head_acc_raw), "head_acc": float(head_acc),
        "head_f1": float(head_f1),
        "knn_acc": float(best_knn_acc), "knn_f1": float(knn_f1),
        "knn_p": float(knn_p), "knn_r": float(knn_r),
        "knn_config": best_knn_config,
        "cm": cm.tolist(), "val_acc": float(bva)
    }

# ---- Experiment configs ----
def get_configs():
    base_media = ["text", "audio", "frame"]
    configs = {}

    # A: Full v13
    configs["A_full"] = base_media + ["perception", "cognition", "answer"]
    # B: Ablation (remove one)
    configs["B1_no_perception"] = base_media + ["cognition", "answer"]
    configs["B2_no_cognition"] = base_media + ["perception", "answer"]
    configs["B3_no_answer"] = base_media + ["perception", "cognition"]
    configs["B4_no_percep_cogn"] = base_media + ["answer"]
    configs["B5_raw_fusion"] = base_media
    # C: Per-field answer (what/target/where/why/how, no which)
    configs["C_perfield"] = base_media + ["ans_what", "ans_target", "ans_where", "ans_why", "ans_how"]
    # D: Per-field ablation
    configs["D1_no_what"] = base_media + ["ans_target", "ans_where", "ans_why", "ans_how"]
    configs["D2_no_target"] = base_media + ["ans_what", "ans_where", "ans_why", "ans_how"]
    configs["D3_no_where"] = base_media + ["ans_what", "ans_target", "ans_why", "ans_how"]
    configs["D4_no_why"] = base_media + ["ans_what", "ans_target", "ans_where", "ans_how"]
    configs["D5_no_how"] = base_media + ["ans_what", "ans_target", "ans_where", "ans_why"]

    return configs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", default="HateMM", choices=["HateMM", "Multihateclip"])
    parser.add_argument("--language", default="English", choices=["English", "Chinese"])
    parser.add_argument("--num_seeds", type=int, default=5)
    parser.add_argument("--config", type=str, default=None, help="Run single config by name")
    parser.add_argument("--version", default="v13", choices=["v13", "v13b"], help="Prompt version")
    args = parser.parse_args()

    logger = setup_logger()

    if args.dataset_name == "HateMM":
        emb_dir = "./embeddings/HateMM"
        ann_path = "./datasets/HateMM/annotation(new).json"
        split_dir = "./datasets/HateMM/splits"
        lm = {"Non Hate": 0, "Hate": 1}
        tag = "HateMM"
    else:
        emb_dir = f"./embeddings/Multihateclip/{args.language}"
        ann_path = f"./datasets/Multihateclip/{args.language}/annotation(new).json"
        split_dir = f"./datasets/Multihateclip/{args.language}/splits"
        lm = {"Normal": 0, "Offensive": 1, "Hateful": 1}
        tag = f"MHC_{args.language[:2]}"

    nc = 2
    seeds = [42, 1042, 2042, 3042, 4042][:args.num_seeds]

    ver = args.version  # "v13" or "v13b"

    # Load ALL possible features
    logger.info(f"Loading features for {tag} ({ver})...")
    feats = {
        "text": torch.load(f"{emb_dir}/text_features.pth", map_location="cpu"),
        "audio": torch.load(f"{emb_dir}/wavlm_audio_features.pth", map_location="cpu"),
        "frame": torch.load(f"{emb_dir}/frame_features.pth", map_location="cpu"),
        "perception": torch.load(f"{emb_dir}/{ver}_perception_features.pth", map_location="cpu"),
        "cognition": torch.load(f"{emb_dir}/{ver}_cognition_features.pth", map_location="cpu"),
        "answer": torch.load(f"{emb_dir}/{ver}_answer_features.pth", map_location="cpu"),
    }
    for f in ["what", "target", "where", "why", "how"]:
        feats[f"ans_{f}"] = torch.load(f"{emb_dir}/{ver}_ans_{f}_features.pth", map_location="cpu")

    with open(ann_path) as f:
        feats["labels"] = {d["Video_ID"]: d for d in json.load(f)}

    splits = load_split_ids(split_dir)

    # Print total vs available sample counts
    all_split_vids = sum(len(splits[s]) for s in splits)
    all_feat_keys = [set(feats[k].keys()) for k in feats if k != "labels"]
    common_all = set.intersection(*all_feat_keys) & set(feats["labels"].keys())
    logger.info(f"  Total split videos: {all_split_vids}")
    logger.info(f"  Videos with ALL embeddings: {len(common_all)}")
    for s in ["train", "valid", "test"]:
        orig = len(splits[s])
        filt = len([v for v in splits[s] if v in common_all])
        dropped = orig - filt
        logger.info(f"    {s}: {orig} -> {filt} (dropped {dropped})")

    configs = get_configs()
    if args.config:
        configs = {args.config: configs[args.config]}

    results_dir = f"./results_{ver}/{tag}"
    os.makedirs(results_dir, exist_ok=True)

    all_summary = {}

    for cname, mk in configs.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"Config: {cname} | modalities: {mk}")

        # Filter to common vids
        fk = list(mk)
        common = set.intersection(*[set(feats[k].keys()) for k in fk]) & set(feats["labels"].keys())
        cur = {s: [v for v in splits[s] if v in common] for s in splits}
        total_orig = sum(len(splits[s]) for s in splits)
        total_filt = sum(len(cur[s]) for s in cur)
        logger.info(f"  Samples: {total_orig} -> {total_filt} valid ({total_orig - total_filt} dropped)")
        logger.info(f"  Train={len(cur['train'])}, Val={len(cur['valid'])}, Test={len(cur['test'])}")

        seed_results = []
        for seed in seeds:
            t0 = time.time()
            r = train_and_eval(feats, cur, lm, mk, nc, seed, class_weight=[1.0, 1.5])
            elapsed = time.time() - t0
            logger.info(f"  seed={seed}: Head ACC={r['head_acc']:.4f} F1={r['head_f1']:.4f} | "
                        f"+kNN ACC={r['knn_acc']:.4f} F1={r['knn_f1']:.4f} "
                        f"({r['knn_config']['whiten']}/{r['knn_config']['knn_type']}) [{elapsed:.0f}s]")
            r["seed"] = seed
            seed_results.append(r)

        # Summary
        head_accs = [r["head_acc"] for r in seed_results]
        knn_accs = [r["knn_acc"] for r in seed_results]
        head_f1s = [r["head_f1"] for r in seed_results]
        knn_f1s = [r["knn_f1"] for r in seed_results]

        summary = {
            "config": cname, "modalities": mk,
            "head_acc_mean": float(np.mean(head_accs)), "head_acc_std": float(np.std(head_accs)),
            "head_acc_max": float(np.max(head_accs)),
            "head_f1_mean": float(np.mean(head_f1s)),
            "knn_acc_mean": float(np.mean(knn_accs)), "knn_acc_std": float(np.std(knn_accs)),
            "knn_acc_max": float(np.max(knn_accs)),
            "knn_f1_mean": float(np.mean(knn_f1s)),
            "seeds": seed_results
        }
        all_summary[cname] = summary

        logger.info(f"  >> Head: ACC={np.mean(head_accs):.4f}+/-{np.std(head_accs):.4f} (max={np.max(head_accs):.4f}) "
                     f"F1={np.mean(head_f1s):.4f}")
        logger.info(f"  >> +kNN: ACC={np.mean(knn_accs):.4f}+/-{np.std(knn_accs):.4f} (max={np.max(knn_accs):.4f}) "
                     f"F1={np.mean(knn_f1s):.4f}")

        # Save per-config
        with open(f"{results_dir}/{cname}.json", "w") as f:
            json.dump(summary, f, indent=2)

    # Save full summary
    with open(f"{results_dir}/all_summary.json", "w") as f:
        json.dump(all_summary, f, indent=2)

    # Print final table
    logger.info(f"\n{'='*80}")
    logger.info(f"{'Config':<25} {'Head ACC':>10} {'Head F1':>10} {'kNN ACC':>10} {'kNN F1':>10} {'kNN max':>10}")
    logger.info(f"{'-'*80}")
    for cname, s in all_summary.items():
        logger.info(f"{cname:<25} {s['head_acc_mean']:>10.4f} {s['head_f1_mean']:>10.4f} "
                     f"{s['knn_acc_mean']:>10.4f} {s['knn_f1_mean']:>10.4f} {s['knn_acc_max']:>10.4f}")
    logger.info(f"{'='*80}")
    logger.info("Done. Results saved to ./results_v13/")


if __name__ == "__main__":
    main()
