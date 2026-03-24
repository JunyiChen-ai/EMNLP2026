"""AppraiseHate Main: Train and evaluate hateful video detection.

Best configuration:
- Prompt: v9 (CAT-informed: appraisal_vector + implicit_meaning + contrastive_readings + stance)
- Text encoder: BERT-base [CLS] max_len=128
- Audio encoder: WavLM-base+
- Vision encoder: ViT
- Fusion: AC-MHGF (Appraisal-Conditioned Multi-Head Gated Fusion) + Uniform Drop 0.15
- Training: AdamW, cosine warmup, EMA, label smoothing

Usage:
    python main.py --num_runs 20
"""

import argparse
import csv
import json
import logging
import os
import random
import copy
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, Dataset

import warnings
warnings.filterwarnings("ignore")

device = "cuda" if torch.cuda.is_available() else "cpu"
MODALITY_KEYS = ["text", "audio", "frame", "t1", "t2"]


# ============================================================
# Logger
# ============================================================

def setup_logger(log_dir="./logs"):
    os.makedirs(log_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"main_{ts}.log")
    logger = logging.getLogger("main")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fh = logging.FileHandler(log_file, encoding="utf-8")
    ch = logging.StreamHandler()
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    fh.setFormatter(fmt)
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.info(f"Log file: {log_file}")
    return logger


# ============================================================
# Dataset
# ============================================================

class HateMM_Dataset(Dataset):
    def __init__(self, video_ids, features, label_map):
        self.video_ids = video_ids
        self.features = features
        self.label_map = label_map

    def __len__(self):
        return len(self.video_ids)

    def __getitem__(self, idx):
        vid = self.video_ids[idx]
        return {
            "text": self.features["text"][vid],
            "audio": self.features["audio"][vid],
            "frame": self.features["frame"][vid],
            "t1": self.features["t1"][vid],
            "t2": self.features["t2"][vid],
            "scores": self.features["scores"][vid],
            "struct": self.features["struct"][vid],
            "label": torch.tensor(
                self.label_map[self.features["labels"][vid]["Label"]],
                dtype=torch.long,
            ),
        }


def collate_fn(batch):
    return {k: torch.stack([b[k] for b in batch]) for k in batch[0]}


# ============================================================
# Model: AC-MHGF (Appraisal-Conditioned Multi-Head Gated Fusion)
# ============================================================

class AC_MHGF(nn.Module):
    """Appraisal-Conditioned Multi-Head Gated Fusion.

    - Per-modality projection + FiLM conditioning from appraisal scores
    - Uniform modality dropout during training
    - Multi-head score-conditioned routing
    - Structured feature integration
    """

    def __init__(
        self,
        num_modalities=5,
        num_heads=4,
        input_dim=768,
        hidden_dim=192,
        num_classes=2,
        score_dim=7,
        struct_dim=36,
        dropout=0.15,
        modality_dropout=0.15,
    ):
        super().__init__()
        self.num_modalities = num_modalities
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.modality_dropout = modality_dropout

        # Per-modality projection
        self.projectors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.LayerNorm(hidden_dim),
            )
            for _ in range(num_modalities)
        ])

        # Score encoder
        self.score_encoder = nn.Sequential(
            nn.Linear(score_dim, 64),
            nn.GELU(),
            nn.Linear(64, 64),
        )

        # FiLM conditioning: scores → per-modality gamma, beta
        self.film = nn.Linear(64, num_modalities * hidden_dim * 2)

        # Structured feature encoder
        self.struct_encoder = nn.Sequential(
            nn.Linear(struct_dim, 64),
            nn.GELU(),
            nn.LayerNorm(64),
        )

        # Multi-head routing
        self.routing_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim + 64, 128),
                nn.GELU(),
                nn.Linear(128, 1),
            )
            for _ in range(num_heads)
        ])

        # Classifier
        classifier_dim = num_heads * hidden_dim + hidden_dim + 64 + 64
        self.classifier = nn.Sequential(
            nn.LayerNorm(classifier_dim),
            nn.Linear(classifier_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(64, num_classes),
        )

    def forward(self, batch, training=False):
        # Score encoding + FiLM parameters
        score_emb = self.score_encoder(batch["scores"])
        film_params = self.film(score_emb).view(
            -1, self.num_modalities, self.hidden_dim, 2
        )

        # Per-modality projection + FiLM conditioning + optional dropout
        refined = []
        for i, (proj, key) in enumerate(zip(self.projectors, MODALITY_KEYS)):
            h = proj(batch[key])
            gamma = film_params[:, i, :, 0]
            beta = film_params[:, i, :, 1]
            h = h * (1 + 0.1 * torch.tanh(gamma)) + 0.1 * beta
            if training and self.modality_dropout > 0:
                mask = (
                    torch.rand(h.size(0), 1, device=h.device) > self.modality_dropout
                ).float()
                h = h * mask
            refined.append(h)

        # Stack modalities
        stacked = torch.stack(refined, dim=1)  # (B, num_mod, hidden)

        # Multi-head routing
        score_expanded = score_emb.unsqueeze(1).expand(-1, self.num_modalities, -1)
        route_input = torch.cat([stacked, score_expanded], dim=-1)

        head_outputs = []
        for routing_head in self.routing_heads:
            weights = torch.softmax(routing_head(route_input).squeeze(-1), dim=1)
            z = (stacked * weights.unsqueeze(-1)).sum(dim=1)
            head_outputs.append(z)

        # Combine heads + mean + score + struct
        mean_pool = stacked.mean(dim=1)
        struct_emb = self.struct_encoder(batch["struct"])

        fusion = torch.cat(head_outputs + [mean_pool, score_emb, struct_emb], dim=-1)
        return self.classifier(fusion)


# ============================================================
# Training utilities
# ============================================================

def get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))

    return LambdaLR(optimizer, lr_lambda)


def load_split_ids(split_dir):
    split_ids = {}
    for split_name in ["train", "valid", "test"]:
        path = os.path.join(split_dir, f"{split_name}.csv")
        with open(path, "r") as f:
            split_ids[split_name] = [row[0] for row in csv.reader(f) if row]
    return split_ids


def evaluate(model, loader, num_classes=2):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(batch, training=False)
            all_preds.extend(logits.argmax(1).cpu().numpy())
            all_labels.extend(batch["label"].cpu().numpy())
    acc = accuracy_score(all_labels, all_preds)
    m_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    w_f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    result = {"acc": acc, "m_f1": m_f1, "w_f1": w_f1}
    # Per-class metrics
    for c in range(num_classes):
        result[f"f1_c{c}"] = f1_score(all_labels, all_preds, pos_label=c, zero_division=0, average="binary") if num_classes == 2 else f1_score((np.array(all_labels) == c).astype(int), (np.array(all_preds) == c).astype(int), zero_division=0)
        result[f"p_c{c}"] = precision_score((np.array(all_labels) == c).astype(int), (np.array(all_preds) == c).astype(int), zero_division=0)
        result[f"r_c{c}"] = recall_score((np.array(all_labels) == c).astype(int), (np.array(all_preds) == c).astype(int), zero_division=0)
    return result


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="AppraiseHate: Train and Evaluate")
    parser.add_argument("--dataset_name", type=str, default="HateMM", choices=["HateMM", "Multihateclip"])
    parser.add_argument("--language", type=str, default="English", choices=["English", "Chinese"])
    parser.add_argument("--num_classes", type=int, default=2, choices=[2, 3])
    parser.add_argument("--num_runs", type=int, default=20, help="Number of runs with different seeds")
    parser.add_argument("--epochs", type=int, default=45)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.02)
    parser.add_argument("--label_smoothing", type=float, default=0.03)
    parser.add_argument("--modality_dropout", type=float, default=0.15)
    parser.add_argument("--ema_decay", type=float, default=0.999)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    args = parser.parse_args()

    logger = setup_logger()
    logger.info(f"Config: {vars(args)}")
    logger.info(f"Device: {device}")

    # Dataset paths
    if args.dataset_name == "HateMM":
        emb_dir = "./embeddings/HateMM"
        ann_path = "./datasets/HateMM/annotation(new).json"
        split_dir = "./datasets/HateMM/splits"
        label_map = {"Non Hate": 0, "Hate": 1}
    else:
        emb_dir = f"./embeddings/Multihateclip/{args.language}"
        ann_path = f"./datasets/Multihateclip/{args.language}/annotation(new).json"
        split_dir = f"./datasets/Multihateclip/{args.language}/splits"
        if args.num_classes == 2:
            label_map = {"Normal": 0, "Offensive": 1, "Hateful": 1}
        else:
            label_map = {"Normal": 0, "Offensive": 1, "Hateful": 2}

    # Load data
    logger.info(f"Loading features from {emb_dir}...")
    features = {
        "text": torch.load(f"{emb_dir}/text_features.pth", map_location="cpu"),
        "audio": torch.load(f"{emb_dir}/wavlm_audio_features.pth", map_location="cpu"),
        "frame": torch.load(f"{emb_dir}/frame_features.pth", map_location="cpu"),
        "t1": torch.load(f"{emb_dir}/v9_t1_features.pth", map_location="cpu"),
        "t2": torch.load(f"{emb_dir}/v9_t2_features.pth", map_location="cpu"),
        "scores": torch.load(f"{emb_dir}/v9_scores.pth", map_location="cpu"),
        "struct": torch.load(f"{emb_dir}/v9_struct_features.pth", map_location="cpu"),
    }
    with open(ann_path) as f:
        features["labels"] = {item["Video_ID"]: item for item in json.load(f)}

    split_ids = load_split_ids(split_dir)

    # Filter to common video IDs
    feat_keys = [k for k in features if k != "labels"]
    common_ids = set.intersection(*[set(features[k].keys()) for k in feat_keys])
    common_ids &= set(features["labels"].keys())
    for s in split_ids:
        split_ids[s] = [v for v in split_ids[s] if v in common_ids]

    struct_dim = features["struct"][list(features["struct"].keys())[0]].shape[0]
    logger.info(
        f"Data: {len(common_ids)} videos, "
        f"train={len(split_ids['train'])}, val={len(split_ids['valid'])}, test={len(split_ids['test'])}, "
        f"struct_dim={struct_dim}"
    )

    # Run experiments
    all_results = []
    for run_idx in range(args.num_runs):
        seed = run_idx * 1000 + 42
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

        # Datasets
        train_ds = HateMM_Dataset(split_ids["train"], features, label_map)
        val_ds = HateMM_Dataset(split_ids["valid"], features, label_map)
        test_ds = HateMM_Dataset(split_ids["test"], features, label_map)

        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, collate_fn=collate_fn)
        test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, collate_fn=collate_fn)

        # Model
        model = AC_MHGF(
            struct_dim=struct_dim,
            num_classes=args.num_classes,
            modality_dropout=args.modality_dropout,
        ).to(device)
        ema_model = copy.deepcopy(model)

        # Optimizer & scheduler
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        total_steps = args.epochs * len(train_loader)
        warmup_steps = args.warmup_epochs * len(train_loader)
        scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
        criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

        # Training loop
        best_val_acc = -1
        best_state = None

        for epoch in range(args.epochs):
            model.train()
            for batch in train_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                optimizer.zero_grad()
                logits = model(batch, training=True)
                loss = criterion(logits, batch["label"])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

                # EMA update
                with torch.no_grad():
                    for p, ep in zip(model.parameters(), ema_model.parameters()):
                        ep.data.mul_(args.ema_decay).add_(p.data, alpha=1 - args.ema_decay)

            # Validation
            val_metrics = evaluate(ema_model, val_loader, args.num_classes)
            if val_metrics["acc"] > best_val_acc:
                best_val_acc = val_metrics["acc"]
                best_state = {k: v.clone() for k, v in ema_model.state_dict().items()}

        # Test
        ema_model.load_state_dict(best_state)
        test_metrics = evaluate(ema_model, test_loader, args.num_classes)
        all_results.append(test_metrics)

        per_class_str = " ".join(f"F1_c{c}={test_metrics[f'f1_c{c}']:.4f}" for c in range(args.num_classes))
        marker = " ***" if test_metrics["acc"] >= 0.90 else (" **" if test_metrics["acc"] >= 0.88 else "")
        logger.info(
            f"Run {run_idx + 1}/{args.num_runs}: "
            f"Acc={test_metrics['acc']:.4f} "
            f"M-F1={test_metrics['m_f1']:.4f} "
            f"{per_class_str}"
            f"{marker}"
        )

    # Summary
    logger.info("")
    logger.info("=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)
    summary_keys = ["acc", "m_f1", "w_f1"] + [f"f1_c{c}" for c in range(args.num_classes)]
    for key in summary_keys:
        vals = [r[key] for r in all_results]
        logger.info(f"  {key:>8}: {np.mean(vals):.4f} ± {np.std(vals):.4f} (max={np.max(vals):.4f})")
    accs = [r["acc"] for r in all_results]
    logger.info(f"  >=0.88: {sum(1 for a in accs if a >= 0.88)}/{args.num_runs}")
    logger.info(f"  >=0.89: {sum(1 for a in accs if a >= 0.89)}/{args.num_runs}")
    logger.info(f"  >=0.90: {sum(1 for a in accs if a >= 0.90)}/{args.num_runs}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
