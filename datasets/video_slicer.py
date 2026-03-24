"""Extract frames from videos. Supports flat files and nested dirs ({vid}/video.ext).
Uses OpenCV first, falls back to PyAV for webm/AV1 that OpenCV can't decode.
"""
import av
import cv2
import numpy as np
import os
import argparse
from tqdm import tqdm


def slice_frames_cv2(video_path, output_dir, num_frames=32):
    """Extract frames using OpenCV."""
    cap = cv2.VideoCapture(str(video_path), cv2.CAP_FFMPEG)
    if not cap.isOpened():
        return 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return 0
    if num_frames <= total_frames:
        seg_size = (total_frames - 1) / num_frames
        selected_ids = [int(np.round(seg_size * i)) for i in range(num_frames)]
    else:
        selected_ids = list(range(total_frames)) * (num_frames // total_frames + 1)
        selected_ids = selected_ids[:num_frames]
    count, saved = 0, 0
    while True:
        success, frame = cap.read()
        if not success:
            break
        if count in selected_ids:
            cv2.imwrite(os.path.join(output_dir, f"frame_{saved+1:03d}.jpg"), frame)
            saved += 1
        count += 1
    cap.release()
    return saved


def slice_frames_pyav(video_path, output_dir, num_frames=32):
    """Extract frames using PyAV (handles webm/AV1)."""
    try:
        container = av.open(str(video_path))
        stream = container.streams.video[0]
        total_frames = stream.frames
        if total_frames <= 0:
            total_frames = sum(1 for _ in container.decode(video=0))
            container.close()
            container = av.open(str(video_path))
        if total_frames <= 0:
            container.close()
            return 0
        if num_frames <= total_frames:
            seg_size = (total_frames - 1) / num_frames
            selected_ids = set(int(np.round(seg_size * i)) for i in range(num_frames))
        else:
            selected_ids = set(range(total_frames))
        saved = 0
        for i, frame in enumerate(container.decode(video=0)):
            if i in selected_ids:
                frame.to_image().save(os.path.join(output_dir, f"frame_{saved+1:03d}.jpg"))
                saved += 1
                if saved >= num_frames:
                    break
        container.close()
        return saved
    except Exception as e:
        print(f"  PyAV error on {video_path}: {e}")
        return 0


def find_video_file(folder):
    """Find video file in a directory."""
    video_extensions = {".mp4", ".webm", ".avi", ".mkv", ".mov"}
    if os.path.isdir(folder):
        for vname in ["video.mp4", "video.webm"]:
            p = os.path.join(folder, vname)
            if os.path.exists(p):
                return p
        for f in os.listdir(folder):
            if os.path.splitext(f)[1].lower() in video_extensions:
                return os.path.join(folder, f)
    return None


def process_folder(input_folder, output_folder, num_frames):
    os.makedirs(output_folder, exist_ok=True)
    video_extensions = {".mp4", ".avi", ".mkv", ".mov", ".webm"}

    # Collect all videos: flat files or nested dirs
    videos = []
    for entry in os.listdir(input_folder):
        entry_path = os.path.join(input_folder, entry)
        if os.path.isfile(entry_path) and os.path.splitext(entry)[1].lower() in video_extensions:
            videos.append((os.path.splitext(entry)[0], entry_path))
        elif os.path.isdir(entry_path):
            vfile = find_video_file(entry_path)
            if vfile:
                videos.append((entry, vfile))

    for vid_name, video_path in tqdm(videos, desc="Extracting frames"):
        frames_dir = os.path.join(output_folder, vid_name)
        if os.path.exists(frames_dir) and len([f for f in os.listdir(frames_dir) if f.endswith('.jpg')]) >= num_frames:
            continue
        os.makedirs(frames_dir, exist_ok=True)
        # Try OpenCV first, fall back to PyAV
        saved = slice_frames_cv2(video_path, frames_dir, num_frames)
        if saved == 0:
            saved = slice_frames_pyav(video_path, frames_dir, num_frames)
        if saved == 0:
            os.rmdir(frames_dir) if not os.listdir(frames_dir) else None


def parse_args():
    parser = argparse.ArgumentParser(description="Extract frames from videos.")
    parser.add_argument("-i", "--input_folder", type=str, required=True)
    parser.add_argument("-o", "--output_folder", type=str, default="frames")
    parser.add_argument("--num_frames", type=int, default=32)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    process_folder(args.input_folder, args.output_folder, args.num_frames)
