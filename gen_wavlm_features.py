"""Extract audio features using WavLM (2025) instead of Wav2Vec.

WavLM captures speech emotion, prosody, and speaker characteristics better than ASR-oriented Wav2Vec.
"""
import json, os, torch, torchaudio
from transformers import AutoModel, AutoFeatureExtractor

device = "cuda"
MODEL_NAME = "microsoft/wavlm-base-plus"


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="HateMM", choices=["HateMM", "Multihateclip", "ImpliHateVid"])
    parser.add_argument("--language", type=str, default="English", choices=["English", "Chinese"])
    args = parser.parse_args()

    if args.dataset_name == "HateMM":
        audio_dir = "./datasets/HateMM/audios"
        ann_path = "./datasets/HateMM/annotation(re).json"
        out_path = "./embeddings/HateMM/wavlm_audio_features.pth"
    elif args.dataset_name == "ImpliHateVid":
        audio_dir = "./datasets/ImpliHateVid/audios"
        ann_path = "./datasets/ImpliHateVid/annotation(new).json"
        out_path = "./embeddings/ImpliHateVid/wavlm_audio_features.pth"
    else:
        audio_dir = f"./datasets/Multihateclip/{args.language}/audios"
        ann_path = f"./datasets/Multihateclip/{args.language}/annotation(new).json"
        out_path = f"./embeddings/Multihateclip/{args.language}/wavlm_audio_features.pth"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device).eval()
    print(f"WavLM: {sum(p.numel() for p in model.parameters())/1e6:.0f}M params")

    with open(ann_path) as f:
        data = json.load(f)

    features = {}
    for i, item in enumerate(data):
        vid = item["Video_ID"]
        audio_path = os.path.join(audio_dir, f"{vid}.wav")
        if not os.path.exists(audio_path):
            print(f"  {vid}: no audio")
            continue

        try:
            waveform, sr = torchaudio.load(audio_path)
            # Resample to 16kHz if needed
            if sr != 16000:
                resampler = torchaudio.transforms.Resample(sr, 16000)
                waveform = resampler(waveform)
            # Mono
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            waveform = waveform.squeeze(0)

            # Truncate to 30s max
            max_samples = 16000 * 30
            if waveform.shape[0] > max_samples:
                waveform = waveform[:max_samples]

            inputs = feature_extractor(waveform.numpy(), sampling_rate=16000, return_tensors="pt", padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                out = model(**inputs)
                # Mean pool over time
                emb = out.last_hidden_state.mean(dim=1).squeeze(0).cpu()

            features[vid] = emb
        except Exception as e:
            print(f"  {vid}: error - {e}")

        if (i + 1) % 100 == 0:
            print(f"Processed {i+1}/{len(data)}")

    torch.save(features, out_path)
    dim = features[list(features.keys())[0]].shape
    print(f"Saved: {len(features)} videos, dim={dim}")


if __name__ == "__main__":
    main()
