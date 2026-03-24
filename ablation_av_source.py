"""Ablation: Compare our audio/frame vs HVGuard's audio/frame embeddings."""
import argparse, csv, json, os, random, copy
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, Dataset
import warnings; warnings.filterwarnings("ignore")
device = "cuda"

from main import AC_MHGF, get_cosine_schedule_with_warmup, load_split_ids, HateMM_Dataset, collate_fn

def run_config(name, features, split_ids, label_map, struct_dim, num_classes=2, num_runs=20):
    feat_keys = [k for k in features if k != "labels"]
    common = set.intersection(*[set(features[k].keys()) for k in feat_keys]) & set(features["labels"].keys())
    cur = {s: [v for v in split_ids[s] if v in common] for s in split_ids}
    print(f"\n=== {name} === (common={len(common)}, train={len(cur['train'])}, test={len(cur['test'])})")

    accs = []
    for ri in range(num_runs):
        seed = ri*1000+42; torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
        trd = HateMM_Dataset(cur["train"], features, label_map)
        vd = HateMM_Dataset(cur["valid"], features, label_map)
        ted = HateMM_Dataset(cur["test"], features, label_map)
        trl = DataLoader(trd, 32, True, collate_fn=collate_fn)
        vl = DataLoader(vd, 64, False, collate_fn=collate_fn)
        tel = DataLoader(ted, 64, False, collate_fn=collate_fn)
        model = AC_MHGF(struct_dim=struct_dim, num_classes=num_classes, modality_dropout=0.15).to(device)
        ema = copy.deepcopy(model)
        opt = optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.02)
        ts = 45*len(trl); ws = 5*len(trl); sch = get_cosine_schedule_with_warmup(opt, ws, ts)
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
    print(f"  Acc: {np.mean(accs):.4f} ± {np.std(accs):.4f} (max={np.max(accs):.4f})")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", required=True, choices=["English", "Chinese"])
    parser.add_argument("--num_runs", type=int, default=20)
    args = parser.parse_args()

    lang = args.language
    emb_dir = f"./embeddings/Multihateclip/{lang}"
    label_map = {"Normal": 0, "Offensive": 1, "Hateful": 1}

    base = {
        "text": torch.load(f"{emb_dir}/text_features.pth", map_location="cpu"),
        "t1": torch.load(f"{emb_dir}/v9_t1_features.pth", map_location="cpu"),
        "t2": torch.load(f"{emb_dir}/v9_t2_features.pth", map_location="cpu"),
        "scores": torch.load(f"{emb_dir}/v9_scores.pth", map_location="cpu"),
        "struct": torch.load(f"{emb_dir}/v9_struct_features.pth", map_location="cpu"),
    }
    with open(f"./datasets/Multihateclip/{lang}/annotation(new).json") as f:
        base["labels"] = {d["Video_ID"]: d for d in json.load(f)}

    split_ids = load_split_ids(f"./datasets/Multihateclip/{lang}/splits")
    struct_dim = base["struct"][list(base["struct"].keys())[0]].shape[0]

    our_audio = torch.load(f"{emb_dir}/wavlm_audio_features.pth", map_location="cpu")
    our_frame = torch.load(f"{emb_dir}/frame_features.pth", map_location="cpu")
    hv_audio = torch.load(f"{emb_dir}/hvguard_audio_features.pth", map_location="cpu")
    hv_frame = torch.load(f"{emb_dir}/hvguard_frame_features.pth", map_location="cpu")

    # Config 1: Our audio + Our frame (current)
    f1 = {**base, "audio": our_audio, "frame": our_frame}
    run_config("Our WavLM + Our ViT", f1, split_ids, label_map, struct_dim, num_runs=args.num_runs)

    # Config 2: HVGuard audio + HVGuard frame
    f2 = {**base, "audio": hv_audio, "frame": hv_frame}
    run_config("HV Wav2Vec + HV ViT", f2, split_ids, label_map, struct_dim, num_runs=args.num_runs)

    # Config 3: HVGuard audio + Our frame
    f3 = {**base, "audio": hv_audio, "frame": our_frame}
    run_config("HV Wav2Vec + Our ViT", f3, split_ids, label_map, struct_dim, num_runs=args.num_runs)

    # Config 4: Our audio + HVGuard frame
    f4 = {**base, "audio": our_audio, "frame": hv_frame}
    run_config("Our WavLM + HV ViT", f4, split_ids, label_map, struct_dim, num_runs=args.num_runs)

if __name__ == "__main__":
    main()
