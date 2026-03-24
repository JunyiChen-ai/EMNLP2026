"""Extract audio from videos using ffmpeg. Supports flat files and nested dirs.
Same approach as HVGuard's preprocess_multihateclip.py.
"""
import os
import subprocess
import argparse
from tqdm import tqdm


VIDEO_EXTENSIONS = {".mp4", ".webm", ".avi", ".mkv", ".mov"}


def find_video_file(folder):
    """Find video file in a directory."""
    if os.path.isdir(folder):
        for vname in ["video.mp4", "video.webm"]:
            p = os.path.join(folder, vname)
            if os.path.exists(p):
                return p
        for f in os.listdir(folder):
            if os.path.splitext(f)[1].lower() in VIDEO_EXTENSIONS:
                return os.path.join(folder, f)
    return None


def extract_audio_ffmpeg(video_path, audio_path):
    """Extract audio using ffmpeg."""
    if os.path.exists(audio_path):
        return True
    try:
        result = subprocess.run(
            ["ffmpeg", "-y", "-i", str(video_path), "-vn", "-acodec", "pcm_s16le",
             "-ar", "16000", "-ac", "1", str(audio_path)],
            capture_output=True, timeout=60
        )
        return result.returncode == 0 and os.path.exists(audio_path)
    except Exception:
        return False


def convert_video_to_audio(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    # Collect all videos: flat files or nested dirs
    videos = []
    for entry in os.listdir(input_folder):
        entry_path = os.path.join(input_folder, entry)
        if os.path.isfile(entry_path) and os.path.splitext(entry)[1].lower() in VIDEO_EXTENSIONS:
            videos.append((os.path.splitext(entry)[0], entry_path))
        elif os.path.isdir(entry_path):
            vfile = find_video_file(entry_path)
            if vfile:
                videos.append((entry, vfile))

    ok, fail = 0, 0
    for vid_name, video_path in tqdm(videos, desc="Extracting audio"):
        audio_path = os.path.join(output_folder, f"{vid_name}.wav")
        if extract_audio_ffmpeg(video_path, audio_path):
            ok += 1
        else:
            fail += 1

    print(f"Done: {ok} success, {fail} failed")


def main():
    parser = argparse.ArgumentParser(description="Extract audio from videos using ffmpeg.")
    parser.add_argument("-i", "--input_folder", type=str, required=True)
    parser.add_argument("-o", "--output_folder", type=str, required=True)
    args = parser.parse_args()
    convert_video_to_audio(args.input_folder, args.output_folder)


if __name__ == "__main__":
    main()
