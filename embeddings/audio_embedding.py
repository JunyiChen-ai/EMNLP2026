from transformers import Wav2Vec2Model, Wav2Vec2Processor
import argparse
import torch
import librosa
from torch import nn
import os
import json

# Extracting audio embeddings(Wav2Vec)
audio_model = "facebook/wav2vec2-base-960h"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Wav2Vec_Model(nn.Module):
    def __init__(self, audio_model, device):
        super(Wav2Vec_Model, self).__init__()
        self.processor = Wav2Vec2Processor.from_pretrained(audio_model)
        self.wav2Vec = Wav2Vec2Model.from_pretrained(audio_model).to(device)

    def forward(self, x):
        # Return the final hidden state
        return self.wav2Vec(x).last_hidden_state

    def process(self, waveform, sampling_rate=16000):
        return self.processor(waveform, sampling_rate=sampling_rate, return_tensors="pt").input_values

    def load_audio(self, file_path, sampling_rate=16000, max_duration=120):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Audio file not found: {file_path}")

        waveform, sr = librosa.load(file_path, sr=sampling_rate)
        max_samples = int(max_duration * sampling_rate)
        if waveform.shape[0] > max_samples:
            waveform = waveform[:max_samples]

        return torch.tensor(waveform)

    def aggregate_features(self, features):
        # Extract CLS token features (first time step)
        return features[:, 0, :]

    def predict(self, audio_path):
        audio_features = []
        with torch.no_grad():
            audio_input = self.load_audio(audio_path)
            input_v = self.process(audio_input).to(self.wav2Vec.device)
            out = self(input_v)
            cls_token_features = self.aggregate_features(
                out)
            audio_features.append(cls_token_features)

        return audio_features[0]


def process_audio_folder(json_path, model):
    with open(json_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    audio_features = {}
    processed_ids = set()

    for item in data:
        video_id = item.get("Video_ID")
        audio_path = item.get("Audio_path")

        if video_id in processed_ids:
            print(f"Skipping duplicate ID: {video_id}")
            continue

        processed_ids.add(video_id)
        print(f"Processing {video_id}...")

        try:
            aggregated_features = model.predict(audio_path)
            audio_features[video_id] = aggregated_features.squeeze(0)
        except Exception as e:
            print(f"Error processing {video_id}: {e}")
            audio_features[video_id] = None

    return audio_features


def save_features_to_pth(features, output_file):
    torch.save(features, output_file)
    print(f"Features saved to {output_file}")


def parse_args():
    parser = argparse.ArgumentParser(description="Extract audio embeddings")
    parser.add_argument("--json", required=True, help="Input data.json path")
    parser.add_argument("--out", required=True, help="Output .pth path")
    return parser.parse_args()


def main():
    args = parse_args()
    json_file = args.json
    output_file = args.out

    model = Wav2Vec_Model(audio_model, device)
    features = process_audio_folder(json_file, model)
    save_features_to_pth(features, output_file)


if __name__ == "__main__":
    main()
