"""Ablation: v12 with 1 sample vs 3 samples on all datasets."""
import json, sys, os, torch, numpy as np, argparse, csv, random, copy
from collections import Counter
sys.path.insert(0, "./embeddings")
from text_embedding import Text_Model
import torch.nn as nn, torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, Dataset
import warnings; warnings.filterwarnings("ignore")

device = "cuda"
MODALITY_KEYS = ["text", "audio", "frame", "t1", "t2"]
STANCES = ["endorse", "report", "condemn", "mock", "unclear"]


def extract_struct(samples, n_use):
    used = samples[:n_use]
    if not used:
        return torch.zeros(len(STANCES) + 4)
    all_stances = []
    has_target = []
    for s in used:
        stance = s.get("speaker_stance", "unclear").strip().lower()
        all_stances.append(STANCES.index(stance) if stance in STANCES else 4)
        t = s.get("target_group", "")
        has_target.append(1.0 if t and t.lower() not in ("none", "n/a", "", "no specific group") else 0.0)
    stance_oh = [0.0] * len(STANCES)
    mc = Counter(all_stances).most_common(1)[0][0]
    stance_oh[mc] = 1.0
    features = list(stance_oh)
    features.append(np.mean(has_target))
    features.append(1.0 if len(set(all_stances)) == 1 else 0.0)
    features.append(sum(1 for s in all_stances if s == 0) / len(all_stances))
    features.append(float(len(used)))
    return torch.tensor(features, dtype=torch.float32)


class DS(Dataset):
    def __init__(self, video_ids, features, label_map):
        self.video_ids = video_ids; self.f = features; self.lm = label_map
    def __len__(self): return len(self.video_ids)
    def __getitem__(self, idx):
        vid = self.video_ids[idx]
        return {k: self.f[k][vid] for k in MODALITY_KEYS} | {
            "struct": self.f["struct"][vid],
            "label": torch.tensor(self.lm[self.f["labels"][vid]["Label"]], dtype=torch.long)}

def collate_fn(batch):
    return {k: torch.stack([b[k] for b in batch]) for k in batch[0]}

class Model(nn.Module):
    def __init__(self, struct_dim=9, num_classes=2, hidden=192, dropout=0.15, mod_drop=0.15):
        super().__init__()
        self.projs = nn.ModuleList([nn.Sequential(nn.Linear(768, hidden), nn.GELU(), nn.Dropout(dropout), nn.LayerNorm(hidden)) for _ in range(5)])
        self.struct_enc = nn.Sequential(nn.Linear(struct_dim, 64), nn.GELU(), nn.LayerNorm(64))
        self.routes = nn.ModuleList([nn.Sequential(nn.Linear(hidden, 128), nn.GELU(), nn.Linear(128, 1)) for _ in range(4)])
        cd = 4 * hidden + hidden + 64
        self.cls = nn.Sequential(nn.LayerNorm(cd), nn.Linear(cd, 256), nn.GELU(), nn.Dropout(dropout), nn.Linear(256, 64), nn.GELU(), nn.Dropout(dropout*0.5), nn.Linear(64, num_classes))
        self.md = mod_drop
    def forward(self, batch, training=False):
        refined = []
        for i, (proj, k) in enumerate(zip(self.projs, MODALITY_KEYS)):
            h = proj(batch[k])
            if training and self.md > 0: h = h * (torch.rand(h.size(0), 1, device=h.device) > self.md).float()
            refined.append(h)
        st = torch.stack(refined, dim=1)
        heads = [((st * torch.softmax(rm(st).squeeze(-1), dim=1).unsqueeze(-1)).sum(dim=1)) for rm in self.routes]
        return self.cls(torch.cat(heads + [st.mean(dim=1), self.struct_enc(batch["struct"])], dim=-1))

def cosine_warmup(opt, ws, ts):
    def f(s):
        if s < ws: return s / max(1, ws)
        return max(0, 0.5 * (1 + np.cos(np.pi * (s - ws) / max(1, ts - ws))))
    return LambdaLR(opt, f)

def run(features, split_ids, label_map, struct_dim, name, num_runs=10):
    accs = []
    for ri in range(num_runs):
        seed = ri*1000+42; torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
        feat_keys = [k for k in features if k != "labels"]
        common = set.intersection(*[set(features[k].keys()) for k in feat_keys]) & set(features["labels"].keys())
        cur = {s: [v for v in split_ids[s] if v in common] for s in split_ids}
        trd = DS(cur["train"], features, label_map); vd = DS(cur["valid"], features, label_map); ted = DS(cur["test"], features, label_map)
        trl = DataLoader(trd, 32, True, collate_fn=collate_fn); vl = DataLoader(vd, 64, False, collate_fn=collate_fn); tel = DataLoader(ted, 64, False, collate_fn=collate_fn)
        model = Model(struct_dim=struct_dim).to(device); ema = copy.deepcopy(model)
        opt = optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.02)
        ts = 45*len(trl); ws = 5*len(trl); sch = cosine_warmup(opt, ws, ts)
        crit = nn.CrossEntropyLoss(label_smoothing=0.03); bva, bst = -1, None
        for ep in range(45):
            model.train()
            for batch in trl:
                batch = {k:v.to(device) for k,v in batch.items()}; opt.zero_grad()
                crit(model(batch, training=True), batch["label"]).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sch.step()
                with torch.no_grad():
                    for p, ep2 in zip(model.parameters(), ema.parameters()): ep2.data.mul_(0.999).add_(p.data, alpha=0.001)
            ema.eval(); ps, ls2 = [], []
            with torch.no_grad():
                for batch in vl:
                    batch = {k:v.to(device) for k,v in batch.items()}
                    ps.extend(ema(batch).argmax(1).cpu().numpy()); ls2.extend(batch["label"].cpu().numpy())
            va = accuracy_score(ls2, ps)
            if va > bva: bva = va; bst = {k:v.clone() for k,v in ema.state_dict().items()}
        ema.load_state_dict(bst); ema.eval(); ps, ls2 = [], []
        with torch.no_grad():
            for batch in tel:
                batch = {k:v.to(device) for k,v in batch.items()}
                ps.extend(ema(batch).argmax(1).cpu().numpy()); ls2.extend(batch["label"].cpu().numpy())
        accs.append(accuracy_score(ls2, ps))
    print(f"  {name}: Acc={np.mean(accs):.4f}±{np.std(accs):.4f} (max={np.max(accs):.4f})")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", default="HateMM", choices=["HateMM", "Multihateclip"])
    parser.add_argument("--language", default="English")
    args = parser.parse_args()

    if args.dataset_name == "HateMM":
        emb_dir = "./embeddings/HateMM"; ann_path = "./datasets/HateMM/annotation(new).json"
        split_dir = "./datasets/HateMM/splits"; label_map = {"Non Hate": 0, "Hate": 1}
        data_path = "./datasets/HateMM/appraise_v12_data.json"
    else:
        emb_dir = f"./embeddings/Multihateclip/{args.language}"
        ann_path = f"./datasets/Multihateclip/{args.language}/annotation(new).json"
        split_dir = f"./datasets/Multihateclip/{args.language}/splits"
        label_map = {"Normal": 0, "Offensive": 1, "Hateful": 1}
        data_path = f"./datasets/Multihateclip/{args.language}/appraise_v12_data.json"

    bert = Text_Model()
    with open(data_path) as f: v12_data = json.load(f)
    v12_map = {d["Video_ID"]: d for d in v12_data}
    with open(ann_path) as f: labels = {d["Video_ID"]: d for d in json.load(f)}

    split_ids = {}
    for s in ["train", "valid", "test"]:
        with open(os.path.join(split_dir, f"{s}.csv")) as f:
            split_ids[s] = [r[0] for r in csv.reader(f) if r]

    base = {
        "text": torch.load(f"{emb_dir}/text_features.pth", map_location="cpu"),
        "audio": torch.load(f"{emb_dir}/wavlm_audio_features.pth", map_location="cpu"),
        "frame": torch.load(f"{emb_dir}/frame_features.pth", map_location="cpu"),
    }

    print(f"\n{args.dataset_name} {args.language}")

    for n_use in [1, 3]:
        t1, t2, struct = {}, {}, {}
        for vid, item in v12_map.items():
            samples = item.get("v12_samples", [])
            s0 = samples[0] if samples else {}
            t1[vid] = bert(s0.get("implicit_meaning", "") or " ").to(device)
            cr = s0.get("contrastive_readings", {})
            t2[vid] = bert(f"Hateful: {cr.get('hateful', '')} Non-hateful: {cr.get('non_hateful', '')}").to(device)
            struct[vid] = extract_struct(samples, n_use).to(device)

        features = {**base, "t1": t1, "t2": t2, "struct": struct, "labels": labels}
        sd = struct[list(struct.keys())[0]].shape[0]
        run(features, split_ids, label_map, sd, f"{n_use}-sample struct", num_runs=10)

    # Also test no struct at all
    t1, t2, struct = {}, {}, {}
    for vid, item in v12_map.items():
        samples = item.get("v12_samples", [])
        s0 = samples[0] if samples else {}
        t1[vid] = bert(s0.get("implicit_meaning", "") or " ").to(device)
        cr = s0.get("contrastive_readings", {})
        t2[vid] = bert(f"Hateful: {cr.get('hateful', '')} Non-hateful: {cr.get('non_hateful', '')}").to(device)
        struct[vid] = torch.zeros(1, device=device)  # dummy

    features = {**base, "t1": t1, "t2": t2, "struct": struct, "labels": labels}
    run(features, split_ids, label_map, 1, "no struct", num_runs=10)


if __name__ == "__main__":
    main()
