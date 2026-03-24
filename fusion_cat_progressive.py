"""CAT-Guided Progressive Fusion (inspired by DEVA, AAAI 2025).

4 appraisal slots: blame/agency, norm-violation, threat/arousal, intentionality.
Each slot is built from text modalities first, then selectively injects audio/frame.

Also includes CAT-Consistency Reliability Gating (MIDAS, NeurIPS 2025):
per-modality reliability predicted from appraisal agreement.

Runs on v12 embeddings (no appraisal_vector scores).
"""
import argparse, csv, json, os, random, copy, logging
from datetime import datetime
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset
import warnings; warnings.filterwarnings("ignore")

device = "cuda"
TEXT_KEYS = ["text", "t1", "t2"]
AV_KEYS = ["audio", "frame"]
ALL_KEYS = TEXT_KEYS + AV_KEYS


def setup_logger():
    os.makedirs("./logs", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    lf = f"./logs/fusion_cat_{ts}.log"
    logger = logging.getLogger("catfusion"); logger.setLevel(logging.INFO); logger.handlers.clear()
    fh = logging.FileHandler(lf, encoding="utf-8"); ch = logging.StreamHandler()
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    fh.setFormatter(fmt); ch.setFormatter(fmt); logger.addHandler(fh); logger.addHandler(ch)
    logger.info(f"Log: {lf}"); return logger


class DS(Dataset):
    def __init__(self, video_ids, features, label_map):
        self.video_ids = video_ids; self.f = features; self.lm = label_map
    def __len__(self): return len(self.video_ids)
    def __getitem__(self, idx):
        vid = self.video_ids[idx]
        return {k: self.f[k][vid] for k in ALL_KEYS} | {
            "struct": self.f["struct"][vid],
            "label": torch.tensor(self.lm[self.f["labels"][vid]["Label"]], dtype=torch.long)}

def collate_fn(batch):
    return {k: torch.stack([b[k] for b in batch]) for k in batch[0]}


class AppraisalSlot(nn.Module):
    """One appraisal slot: builds from text, then gated injection of audio/frame."""
    def __init__(self, hidden=128, text_dim=128, av_dim=128):
        super().__init__()
        # Text aggregation
        self.text_attn = nn.Sequential(nn.Linear(text_dim, 64), nn.Tanh(), nn.Linear(64, 1))
        # AV injection gates
        self.audio_gate = nn.Sequential(nn.Linear(hidden + av_dim, 64), nn.ReLU(), nn.Linear(64, 1), nn.Sigmoid())
        self.frame_gate = nn.Sequential(nn.Linear(hidden + av_dim, 64), nn.ReLU(), nn.Linear(64, 1), nn.Sigmoid())
        self.combine = nn.Linear(hidden + av_dim * 2, hidden)
        self.ln = nn.LayerNorm(hidden)

    def forward(self, text_feats, audio_feat, frame_feat):
        # text_feats: (B, 3, hidden) — text, T1, T2
        # Attention over text modalities for this slot
        weights = torch.softmax(self.text_attn(text_feats).squeeze(-1), dim=1)  # (B, 3)
        slot = (text_feats * weights.unsqueeze(-1)).sum(dim=1)  # (B, hidden)

        # Gated injection of audio/frame
        ag = self.audio_gate(torch.cat([slot, audio_feat], dim=-1))  # (B, 1)
        fg = self.frame_gate(torch.cat([slot, frame_feat], dim=-1))  # (B, 1)
        injected_audio = ag * audio_feat
        injected_frame = fg * frame_feat

        out = self.combine(torch.cat([slot, injected_audio, injected_frame], dim=-1))
        return self.ln(F.gelu(out))


class CATProgressiveFusion(nn.Module):
    """CAT-Guided Progressive Fusion with 4 appraisal slots + reliability gating."""
    def __init__(self, num_slots=4, hidden=128, struct_dim=9, num_classes=2, dropout=0.15):
        super().__init__()
        # Per-modality projection
        self.text_projs = nn.ModuleList([
            nn.Sequential(nn.Linear(768, hidden), nn.GELU(), nn.Dropout(dropout), nn.LayerNorm(hidden))
            for _ in range(3)])
        self.audio_proj = nn.Sequential(nn.Linear(768, hidden), nn.GELU(), nn.Dropout(dropout), nn.LayerNorm(hidden))
        self.frame_proj = nn.Sequential(nn.Linear(768, hidden), nn.GELU(), nn.Dropout(dropout), nn.LayerNorm(hidden))

        # 4 appraisal slots
        self.slots = nn.ModuleList([AppraisalSlot(hidden, hidden, hidden) for _ in range(num_slots)])

        # Reliability gating (MIDAS-inspired)
        self.audio_reliability = nn.Sequential(nn.Linear(hidden * num_slots, 64), nn.ReLU(), nn.Linear(64, 1), nn.Sigmoid())
        self.frame_reliability = nn.Sequential(nn.Linear(hidden * num_slots, 64), nn.ReLU(), nn.Linear(64, 1), nn.Sigmoid())

        # Struct encoder
        self.struct_enc = nn.Sequential(nn.Linear(struct_dim, 64), nn.GELU(), nn.LayerNorm(64))

        # Classifier
        cd = hidden * num_slots + hidden * 3 + 64  # slots + text residuals + struct
        self.classifier = nn.Sequential(
            nn.LayerNorm(cd), nn.Linear(cd, 256), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(256, 64), nn.GELU(), nn.Dropout(dropout * 0.5),
            nn.Linear(64, num_classes))

    def forward(self, batch, training=False):
        # Project all modalities
        text_feats = torch.stack([proj(batch[k]) for proj, k in zip(self.text_projs, TEXT_KEYS)], dim=1)  # (B, 3, H)
        audio = self.audio_proj(batch["audio"])  # (B, H)
        frame = self.frame_proj(batch["frame"])  # (B, H)

        # Run 4 appraisal slots
        slot_outs = [slot(text_feats, audio, frame) for slot in self.slots]
        slot_cat = torch.cat(slot_outs, dim=-1)  # (B, 4*H)

        # Reliability gating — suppress audio/frame if inconsistent with appraisal slots
        ar = self.audio_reliability(slot_cat)  # (B, 1)
        fr = self.frame_reliability(slot_cat)  # (B, 1)
        reliable_audio = ar * audio
        reliable_frame = fr * frame

        # Text residuals (always trusted)
        text_pool = text_feats.mean(dim=1)  # (B, H)

        # Struct
        struct = self.struct_enc(batch["struct"])

        # Combine
        combined = torch.cat([slot_cat, text_pool, reliable_audio, reliable_frame, struct], dim=-1)
        return self.classifier(combined)


def cosine_warmup(opt, ws, ts):
    def f(s):
        if s < ws: return s / max(1, ws)
        return max(0, 0.5 * (1 + np.cos(np.pi * (s - ws) / max(1, ts - ws))))
    return LambdaLR(opt, f)


def load_split_ids(split_dir):
    s = {}
    for n in ["train", "valid", "test"]:
        with open(os.path.join(split_dir, f"{n}.csv")) as f:
            s[n] = [r[0] for r in csv.reader(f) if r]
    return s


def run_experiment(features, split_ids, label_map, struct_dim, num_classes, logger, name, num_runs=20, epochs=45):
    feat_keys = [k for k in features if k != "labels"]
    common = set.intersection(*[set(features[k].keys()) for k in feat_keys]) & set(features["labels"].keys())
    cur = {s: [v for v in split_ids[s] if v in common] for s in split_ids}
    logger.info(f"=== {name} === common={len(common)}, train={len(cur['train'])}, val={len(cur['valid'])}, test={len(cur['test'])}")

    accs, mf1s = [], []
    for ri in range(num_runs):
        seed = ri * 1000 + 42; torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
        trd = DS(cur["train"], features, label_map); vd = DS(cur["valid"], features, label_map); ted = DS(cur["test"], features, label_map)
        trl = DataLoader(trd, 32, True, collate_fn=collate_fn); vl = DataLoader(vd, 64, False, collate_fn=collate_fn); tel = DataLoader(ted, 64, False, collate_fn=collate_fn)
        model = CATProgressiveFusion(struct_dim=struct_dim, num_classes=num_classes).to(device)
        ema = copy.deepcopy(model)
        opt = optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.02)
        ts = epochs * len(trl); ws = 5 * len(trl); sch = cosine_warmup(opt, ws, ts)
        crit = nn.CrossEntropyLoss(label_smoothing=0.03); bva, bst = -1, None
        for ep in range(epochs):
            model.train()
            for batch in trl:
                batch = {k: v.to(device) for k, v in batch.items()}; opt.zero_grad()
                crit(model(batch, training=True), batch["label"]).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sch.step()
                with torch.no_grad():
                    for p, ep2 in zip(model.parameters(), ema.parameters()): ep2.data.mul_(0.999).add_(p.data, alpha=0.001)
            ema.eval(); ps, ls2 = [], []
            with torch.no_grad():
                for batch in vl:
                    batch = {k: v.to(device) for k, v in batch.items()}
                    ps.extend(ema(batch).argmax(1).cpu().numpy()); ls2.extend(batch["label"].cpu().numpy())
            va = accuracy_score(ls2, ps)
            if va > bva: bva = va; bst = {k: v.clone() for k, v in ema.state_dict().items()}
        ema.load_state_dict(bst); ema.eval(); ps, ls2 = [], []
        with torch.no_grad():
            for batch in tel:
                batch = {k: v.to(device) for k, v in batch.items()}
                ps.extend(ema(batch).argmax(1).cpu().numpy()); ls2.extend(batch["label"].cpu().numpy())
        acc = accuracy_score(ls2, ps); mf1 = f1_score(ls2, ps, average="macro", zero_division=0)
        accs.append(acc); mf1s.append(mf1)
        mk = " ***" if acc >= 0.90 else (" **" if acc >= 0.85 else "")
        logger.info(f"  Run {ri+1}/{num_runs}: Acc={acc:.4f} M-F1={mf1:.4f}{mk}")

    logger.info(f"  SUMMARY: Acc={np.mean(accs):.4f}±{np.std(accs):.4f} (max={np.max(accs):.4f})")
    logger.info(f"  M-F1: {np.mean(mf1s):.4f}±{np.std(mf1s):.4f}")
    logger.info(f"  >=0.85: {sum(1 for a in accs if a>=0.85)}, >=0.88: {sum(1 for a in accs if a>=0.88)}, >=0.90: {sum(1 for a in accs if a>=0.90)}")
    return np.mean(accs), np.max(accs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", default="HateMM", choices=["HateMM", "Multihateclip"])
    parser.add_argument("--language", default="English", choices=["English", "Chinese"])
    parser.add_argument("--num_runs", type=int, default=20)
    args = parser.parse_args()
    logger = setup_logger()

    if args.dataset_name == "HateMM":
        emb_dir = "./embeddings/HateMM"; ann_path = "./datasets/HateMM/annotation(new).json"
        split_dir = "./datasets/HateMM/splits"; label_map = {"Non Hate": 0, "Hate": 1}
    else:
        emb_dir = f"./embeddings/Multihateclip/{args.language}"
        ann_path = f"./datasets/Multihateclip/{args.language}/annotation(new).json"
        split_dir = f"./datasets/Multihateclip/{args.language}/splits"
        label_map = {"Normal": 0, "Offensive": 1, "Hateful": 1}

    features = {
        "text": torch.load(f"{emb_dir}/text_features.pth", map_location="cpu"),
        "audio": torch.load(f"{emb_dir}/wavlm_audio_features.pth", map_location="cpu"),
        "frame": torch.load(f"{emb_dir}/frame_features.pth", map_location="cpu"),
        "t1": torch.load(f"{emb_dir}/v12_t1_features.pth", map_location="cpu"),
        "t2": torch.load(f"{emb_dir}/v12_t2_features.pth", map_location="cpu"),
        "struct": torch.load(f"{emb_dir}/v12_struct_features.pth", map_location="cpu"),
    }
    with open(ann_path) as f: features["labels"] = {d["Video_ID"]: d for d in json.load(f)}
    split_ids = load_split_ids(split_dir)
    struct_dim = features["struct"][list(features["struct"].keys())[0]].shape[0]

    run_experiment(features, split_ids, label_map, struct_dim, 2, logger,
                   f"{args.dataset_name} {args.language} CAT-Progressive", args.num_runs)


if __name__ == "__main__":
    main()
