"""
Full preprocessing pipeline for ImpliHateVid dataset.
1. Create annotation JSON + split CSVs
2. Extract frames from videos
3. Build quad images
4. Extract audio (wav)
5. Run v13b LLM (Harmful/Normal prompt)
6. Generate embeddings (text, audio, frame, v13b answer fields)

Label mapping: Non Hate -> 0, Explicit Hate + Implicit Hate -> 1 (binary)
"""
import json, os, csv, subprocess, glob
import pandas as pd
from pathlib import Path

BASE = "/home/junyi/EMNLP2026"
DS_DIR = f"{BASE}/datasets/ImpliHateVid"
EMB_DIR = f"{BASE}/embeddings/ImpliHateVid"
os.makedirs(EMB_DIR, exist_ok=True)

def get_label(vid):
    if vid.startswith("EX_") or vid.startswith("IM_"):
        return "Hateful"
    return "Normal"

def get_video_path(vid):
    if vid.startswith("EX_"):
        folder = f"{DS_DIR}/Explicit_Hate_Videos"
    elif vid.startswith("IM_"):
        folder = f"{DS_DIR}/Implicit_Hate_Videos"
    else:
        folder = f"{DS_DIR}/Non_Hate_Videos"
    for ext in ['.mp4', '.webm', '.mkv', '.avi', '.mov']:
        p = os.path.join(folder, vid + ext)
        if os.path.exists(p):
            return p
    # Try without ID prefix match
    for f in os.listdir(folder):
        if f.startswith(vid):
            return os.path.join(folder, f)
    return None

# ====== Step 1: Annotation + Splits ======
print("Step 1: Creating annotation JSON and split CSVs...")
all_items = []
for split_name, xlsx_name in [("train", "Train_videos.xlsx"), ("valid", "Val_videos.xlsx"), ("test", "Test_videos.xlsx")]:
    df = pd.read_excel(f"{DS_DIR}/{xlsx_name}")
    for _, row in df.iterrows():
        vid = row['Video_ID']
        label = get_label(vid)
        vpath = get_video_path(vid)
        all_items.append({
            "Video_ID": vid,
            "Label": label,
            "Title": "",
            "Transcript": "",
            "split": split_name,
            "video_path": vpath or "",
        })

# Save annotation
ann = [{"Video_ID": it["Video_ID"], "Label": it["Label"], "Title": it["Title"], "Transcript": it["Transcript"]} for it in all_items]
with open(f"{DS_DIR}/annotation(new).json", "w") as f:
    json.dump(ann, f, indent=2)
print(f"  Annotation: {len(ann)} items")

# Save splits
os.makedirs(f"{DS_DIR}/splits", exist_ok=True)
for split_name in ["train", "valid", "test"]:
    vids = [it["Video_ID"] for it in all_items if it["split"] == split_name]
    with open(f"{DS_DIR}/splits/{split_name}.csv", "w", newline="") as f:
        w = csv.writer(f)
        for v in vids:
            w.writerow([v])
    print(f"  {split_name}: {len(vids)}")

# Label distribution
from collections import Counter
labels = Counter(it["Label"] for it in all_items)
print(f"  Labels: {dict(labels)}")

# ====== Step 2: Extract frames ======
print("\nStep 2: Extracting frames...")
frames_dir = f"{DS_DIR}/frames"
os.makedirs(frames_dir, exist_ok=True)
missing_video = 0
for it in all_items:
    vid = it["Video_ID"]
    vpath = it["video_path"]
    out_dir = f"{frames_dir}/{vid}"
    if os.path.isdir(out_dir) and len(os.listdir(out_dir)) > 0:
        continue
    if not vpath or not os.path.exists(vpath):
        missing_video += 1
        continue
    os.makedirs(out_dir, exist_ok=True)
    # Extract 1 frame per second using ffmpeg
    cmd = f'ffmpeg -i "{vpath}" -vf "fps=1" -q:v 2 "{out_dir}/frame_%04d.jpg" -y -loglevel error'
    subprocess.run(cmd, shell=True, timeout=60)
print(f"  Done. Missing videos: {missing_video}")

# ====== Step 3: Build quad images ======
print("\nStep 3: Building quad images...")
quad_dir = f"{DS_DIR}/quad"
os.makedirs(quad_dir, exist_ok=True)

from PIL import Image
def build_quad(frame_files, out_path, size=(512, 512)):
    """Combine 4 frames into a 2x2 quad image."""
    quad = Image.new('RGB', (size[0]*2, size[1]*2))
    for i, fp in enumerate(frame_files[:4]):
        img = Image.open(fp).resize(size)
        x = (i % 2) * size[0]
        y = (i // 2) * size[1]
        quad.paste(img, (x, y))
    quad.save(out_path, quality=85)

built = 0
for it in all_items:
    vid = it["Video_ID"]
    fdir = f"{frames_dir}/{vid}"
    qdir = f"{quad_dir}/{vid}"
    if os.path.isdir(qdir) and len(os.listdir(qdir)) > 0:
        continue
    if not os.path.isdir(fdir):
        continue
    frame_files = sorted(glob.glob(f"{fdir}/*.jpg"))
    if len(frame_files) < 4:
        continue
    os.makedirs(qdir, exist_ok=True)
    # Build quads from consecutive groups of 4 frames
    for qi in range(0, len(frame_files), 4):
        batch = frame_files[qi:qi+4]
        if len(batch) == 4:
            build_quad(batch, f"{qdir}/quad_{qi//4:03d}.jpg")
    built += 1
print(f"  Built quads for {built} videos")

# ====== Step 4: Extract audio ======
print("\nStep 4: Extracting audio...")
audio_dir = f"{DS_DIR}/audios"
os.makedirs(audio_dir, exist_ok=True)
for it in all_items:
    vid = it["Video_ID"]
    vpath = it["video_path"]
    wav_path = f"{audio_dir}/{vid}.wav"
    if os.path.exists(wav_path):
        continue
    if not vpath or not os.path.exists(vpath):
        continue
    cmd = f'ffmpeg -i "{vpath}" -ac 1 -ar 16000 "{wav_path}" -y -loglevel error'
    try:
        subprocess.run(cmd, shell=True, timeout=60)
    except:
        pass
audio_count = len([f for f in os.listdir(audio_dir) if f.endswith('.wav')])
print(f"  Extracted {audio_count} audio files")

print("\nPreprocessing steps 1-4 done.")
print("Next: Run v13b LLM, then generate embeddings.")
