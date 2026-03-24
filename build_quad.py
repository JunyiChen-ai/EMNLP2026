import argparse
import math
import os
from pathlib import Path

from PIL import Image
from tqdm import tqdm


DEFAULT_DATASETS = [
    "datasets/HateMM",
    "datasets/Multihateclip/Chinese",
    "datasets/Multihateclip/English",
]


def list_frame_files(video_dir: Path):
    return sorted(
        [
            path for path in video_dir.iterdir()
            if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        ]
    )


def build_quad_image(frame_paths):
    images = [Image.open(path).convert("RGB") for path in frame_paths]
    min_width = min(image.width for image in images)
    min_height = min(image.height for image in images)
    images = [image.resize((min_width, min_height)) for image in images]

    canvas = Image.new("RGB", (min_width * 2, min_height * 2))
    canvas.paste(images[0], (0, 0))
    canvas.paste(images[1], (min_width, 0))
    canvas.paste(images[2], (0, min_height))
    canvas.paste(images[3], (min_width, min_height))
    return canvas


def process_video_dir(video_dir: Path, output_dir: Path):
    frame_files = list_frame_files(video_dir)
    if not frame_files:
        return False, "no_frames"

    num_quads = math.ceil(len(frame_files) / 4)
    output_dir.mkdir(parents=True, exist_ok=True)

    existing_quads = sorted(output_dir.glob("quad_*.jpg"))
    if len(existing_quads) == num_quads:
        return True, "skipped"

    for quad_index in range(num_quads):
        start = quad_index * 4
        quad_frames = frame_files[start:start + 4]
        while len(quad_frames) < 4:
            quad_frames.append(quad_frames[-1])

        quad_image = build_quad_image(quad_frames)
        quad_image.save(output_dir / f"quad_{quad_index + 1:03d}.jpg", "JPEG")

    return True, "processed"


def process_dataset(dataset_root: Path):
    frames_root = dataset_root / "frames"
    quad_root = dataset_root / "quad"

    if not frames_root.exists():
        print(f"Skipping {dataset_root}: missing {frames_root}")
        return

    video_dirs = sorted([path for path in frames_root.iterdir() if path.is_dir()])
    if not video_dirs:
        print(f"Skipping {dataset_root}: no video frame folders found")
        return

    processed = 0
    skipped = 0
    failed = 0

    for video_dir in tqdm(video_dirs, desc=f"Processing {dataset_root}", unit="video"):
        ok, status = process_video_dir(video_dir, quad_root / video_dir.name)
        if ok and status == "processed":
            processed += 1
        elif ok and status == "skipped":
            skipped += 1
        else:
            failed += 1

    print(f"{dataset_root}: processed={processed}, skipped={skipped}, failed={failed}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Combine every 4 frames into one 2x2 quad image."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=DEFAULT_DATASETS,
        help="Dataset roots relative to the project root.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    for dataset in args.datasets:
        process_dataset(Path(dataset))


if __name__ == "__main__":
    main()
