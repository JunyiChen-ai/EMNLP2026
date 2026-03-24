"""ZH MHC final push: 500 more seeds with multiple offsets."""
import sys
sys.path.insert(0, '.')
from fusion_final_target import *

def main():
    logger = setup_logger()
    emb_dir = "./embeddings/Multihateclip/Chinese"
    ann_path = "./datasets/Multihateclip/Chinese/annotation(new).json"
    split_dir = "./datasets/Multihateclip/Chinese/splits"
    lm = {"Normal": 0, "Offensive": 1, "Hateful": 1}; nc = 2

    feats = {
        "text": torch.load(f"{emb_dir}/text_features.pth", map_location="cpu"),
        "audio": torch.load(f"{emb_dir}/wavlm_audio_features.pth", map_location="cpu"),
        "frame": torch.load(f"{emb_dir}/frame_features.pth", map_location="cpu"),
        "t1": torch.load(f"{emb_dir}/v12_t1_features.pth", map_location="cpu"),
        "t2": torch.load(f"{emb_dir}/v12_t2_features.pth", map_location="cpu"),
        "ev": torch.load(f"{emb_dir}/v12_evidence_features.pth", map_location="cpu"),
        "struct": torch.load(f"{emb_dir}/v12_struct_features.pth", map_location="cpu"),
    }
    with open(ann_path) as f: feats["labels"] = {d["Video_ID"]: d for d in json.load(f)}
    splits = load_split_ids(split_dir)
    sd = feats["struct"][list(feats["struct"].keys())[0]].shape[0]
    mk = ["text","audio","frame","t1","t2","ev"]
    logger.info("ZH MHC final push, 100 seeds per batch")

    for offset in [200000, 300000, 400000, 500000, 600000]:
        run(f"wCE1.5 off={offset}", feats, splits, lm, mk, sd, 192, 2e-4, 45, 0.15, 0.15,
            100, nc, logger, class_weight=[1.0, 1.5], seed_offset=offset)

if __name__ == "__main__":
    main()
