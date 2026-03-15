import os
import json
import argparse
import torch
from PIL import Image
import torch.nn as nn
from transformers import ViTModel, ViTFeatureExtractor

# Extract video frame embeddings(Vit)
vision_model = "google/vit-base-patch16-224"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vit_model = ViTModel.from_pretrained(vision_model).to(device)
feature_extractor = ViTFeatureExtractor.from_pretrained(vision_model)

class FrameFeatureExtractor(nn.Module):
    def __init__(self, vit_model):
        super(FrameFeatureExtractor, self).__init__()
        self.vit = vit_model

    def forward(self, frame):
        frame = frame.unsqueeze(0)
        with torch.no_grad():
            outputs = self.vit(pixel_values=frame)
            # Extract pool_output (CLS token representation)
            pool_output = outputs.pooler_output
        return pool_output

def process_single_video(video_id, frames_path, feature_extractor, frame_extractor, frame_interval):
    frame_features = []

    frame_files = sorted(os.listdir(frames_path))
    selected_frames = frame_files[::frame_interval]

    for frame_file in selected_frames:
        frame_path = os.path.join(frames_path, frame_file)
        if frame_file.lower().endswith(('png', 'jpg', 'jpeg')):
            image = Image.open(frame_path).convert("RGB")
            inputs = feature_extractor(images=image, return_tensors="pt")
            pixel_values = inputs["pixel_values"].squeeze(0).to(device)
            feature = frame_extractor(pixel_values)
            torch.cuda.empty_cache()
            frame_features.append(feature)

    # Aggregate features (average pooling)
    frame_features = torch.cat(frame_features, dim=0)
    video_representation = torch.mean(
        frame_features, dim=0)

    return video_representation


def process_dataset(json_path, vit_model, feature_extractor, save_path):
    with open(json_path, 'r') as f:
        dataset = json.load(f)

    frame_extractor = FrameFeatureExtractor(vit_model).to(device)

    video_features = {}

    for entry in dataset:
        video_id = entry['Video_ID']
        frames_path = entry['Frames_path']

        if not os.path.exists(frames_path):
            print(
                f"Warning: Frames path does not exist for Video_ID {video_id}")
            continue

        print(f"Processing {video_id}...")
        video_representation = process_single_video(
            video_id, frames_path, feature_extractor, frame_extractor, frame_interval=2)
        video_features[video_id] = video_representation

    torch.save(video_features, save_path)
    print(f"All features saved to {save_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Extract frame embeddings")
    parser.add_argument("--json", required=True, help="Input data.json path")
    parser.add_argument("--out", required=True, help="Output .pth path")
    return parser.parse_args()


def main():
    args = parse_args()
    json_file = args.json
    output_file = args.out
    process_dataset(json_file, vit_model, feature_extractor, output_file)


if __name__ == "__main__":
    main()
