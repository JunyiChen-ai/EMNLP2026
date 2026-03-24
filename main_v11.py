"""Train/evaluate with AppraiseHate v11 features without touching main.py."""

import argparse
import copy
import json
import random
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from main import (
    AC_MHGF,
    HateMM_Dataset,
    collate_fn,
    device,
    evaluate,
    get_cosine_schedule_with_warmup,
    load_split_ids,
    setup_logger,
)

warnings.filterwarnings("ignore")


def main():
    parser = argparse.ArgumentParser(description="AppraiseHate v11: Train and Evaluate")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="HateMM",
        choices=["HateMM", "Multihateclip"],
    )
    parser.add_argument(
        "--language",
        type=str,
        default="English",
        choices=["English", "Chinese"],
    )
    parser.add_argument("--num_classes", type=int, default=2, choices=[2, 3])
    parser.add_argument("--num_runs", type=int, default=20)
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

    logger.info(f"Loading v11 features from {emb_dir}...")
    features = {
        "text": torch.load(f"{emb_dir}/text_features.pth", map_location="cpu"),
        "audio": torch.load(f"{emb_dir}/wavlm_audio_features.pth", map_location="cpu"),
        "frame": torch.load(f"{emb_dir}/frame_features.pth", map_location="cpu"),
        "t1": torch.load(f"{emb_dir}/v11_t1_features.pth", map_location="cpu"),
        "t2": torch.load(f"{emb_dir}/v11_t2_features.pth", map_location="cpu"),
        "scores": torch.load(f"{emb_dir}/v11_scores.pth", map_location="cpu"),
        "struct": torch.load(f"{emb_dir}/v11_struct_features.pth", map_location="cpu"),
    }
    with open(ann_path) as f:
        features["labels"] = {item["Video_ID"]: item for item in json.load(f)}

    split_ids = load_split_ids(split_dir)
    feat_keys = [key for key in features if key != "labels"]
    common_ids = set.intersection(*[set(features[key].keys()) for key in feat_keys])
    common_ids &= set(features["labels"].keys())
    for split_name in split_ids:
        split_ids[split_name] = [vid for vid in split_ids[split_name] if vid in common_ids]

    struct_dim = features["struct"][list(features["struct"].keys())[0]].shape[0]
    logger.info(
        f"Data: {len(common_ids)} videos, "
        f"train={len(split_ids['train'])}, val={len(split_ids['valid'])}, "
        f"test={len(split_ids['test'])}, struct_dim={struct_dim}"
    )

    all_results = []
    for run_idx in range(args.num_runs):
        seed = run_idx * 1000 + 42
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

        train_ds = HateMM_Dataset(split_ids["train"], features, label_map)
        val_ds = HateMM_Dataset(split_ids["valid"], features, label_map)
        test_ds = HateMM_Dataset(split_ids["test"], features, label_map)

        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
        )
        val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, collate_fn=collate_fn)
        test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, collate_fn=collate_fn)

        model = AC_MHGF(
            struct_dim=struct_dim,
            num_classes=args.num_classes,
            modality_dropout=args.modality_dropout,
        ).to(device)
        ema_model = copy.deepcopy(model)

        optimizer = optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
        total_steps = args.epochs * len(train_loader)
        warmup_steps = args.warmup_epochs * len(train_loader)
        scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
        criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

        best_val_acc = -1.0
        best_state = None

        for epoch in range(args.epochs):
            model.train()
            for batch in train_loader:
                batch = {key: value.to(device) for key, value in batch.items()}
                optimizer.zero_grad()
                logits = model(batch, training=True)
                loss = criterion(logits, batch["label"])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

                with torch.no_grad():
                    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
                        ema_param.data.mul_(args.ema_decay).add_(param.data, alpha=1 - args.ema_decay)

            val_result = evaluate(ema_model, val_loader, num_classes=args.num_classes)
            if val_result["acc"] > best_val_acc:
                best_val_acc = val_result["acc"]
                best_state = copy.deepcopy(ema_model.state_dict())

            logger.info(
                f"Run {run_idx + 1}/{args.num_runs} "
                f"Epoch {epoch + 1}/{args.epochs} "
                f"val_acc={val_result['acc']:.4f} val_mf1={val_result['m_f1']:.4f}"
            )

        if best_state is not None:
            ema_model.load_state_dict(best_state)
        test_result = evaluate(ema_model, test_loader, num_classes=args.num_classes)
        all_results.append(test_result)
        logger.info(
            f"Run {run_idx + 1}/{args.num_runs} "
            f"test_acc={test_result['acc']:.4f} test_mf1={test_result['m_f1']:.4f}"
        )

    keys = all_results[0].keys()
    mean_result = {key: float(np.mean([res[key] for res in all_results])) for key in keys}
    std_result = {key: float(np.std([res[key] for res in all_results])) for key in keys}

    logger.info("===== Final Results (v11) =====")
    for key in keys:
        logger.info(f"{key}: {mean_result[key]:.4f} +/- {std_result[key]:.4f}")


if __name__ == "__main__":
    main()
