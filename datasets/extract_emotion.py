import os
import json
import re
from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess

# Extract audio transcripts and prosodic emotions (Fun-ASR)
def load_model(model_dir, device="cuda:0"):
    model = AutoModel(
        model=model_dir,
        vad_model="fsmn-vad",
        vad_kwargs={"max_single_segment_time": 30000},
        device=device,
        hub="hf",
    )
    return model

# Perform speech recognition
def recognize_speech(model, audio_path, language="auto", use_itn=True, batch_size_s=60, merge_vad=True, merge_length_s=15):
    result = model.generate(
        input=audio_path,
        cache={},
        language=language, 
        use_itn=use_itn,
        batch_size_s=batch_size_s,
        merge_vad=merge_vad,
        merge_length_s=merge_length_s,
    )
    return result

# Extract emotion labels from the recognition results
def extract_emotion(result):
    emotions = []
    for item in result:
        text = item.get('text', '')
        matches = re.findall(r"<\|([A-Z_]+)\|>", text)
        filtered = [token for token in matches if token != "EMO_UNKNOWN"]
        deduped = list(dict.fromkeys(filtered))
        emotion = ";".join(deduped)
        emotions.append(emotion)
    return emotions

def process_audio_folder(model, folder_path, output_json, annotation_path, frames_path):
    wav_files = [f for f in os.listdir(folder_path) if f.lower().endswith(".wav")]
    with open(annotation_path, 'r', encoding='utf-8') as json_file:
        annotation_data = json.load(json_file)

    metadata = {
        item["Video_ID"]: item for item in annotation_data if item.get("Video_ID")
    }
    results = []

    for wav_file in wav_files:
        audio_path = os.path.join(folder_path, wav_file)
        video_id = os.path.splitext(wav_file)[0]
        if video_id not in metadata:
            print(f"Skipping {video_id}: missing in annotation(new).json")
            continue

        print(f"Processing {video_id}...")
        res = recognize_speech(model, audio_path)
        emotions = extract_emotion(res)
        transcript = rich_transcription_postprocess(res[0]["text"])
        emotion = emotions[0] if emotions and emotions[0] else "Unknown"
        source_item = metadata[video_id]
        result_item = dict(source_item)
        result_item["Transcript"] = transcript
        result_item["Emotion"] = emotion
        result_item["Frames_path"] = os.path.join(frames_path, video_id)
        result_item["Audio_path"] = audio_path
        results.append(result_item)

    with open(output_json, 'w', encoding='utf-8') as json_file:
        json.dump(results, json_file, ensure_ascii=False, indent=4)

def parse_args():
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract emotion from videos in a folder.")
    parser.add_argument("-i", "--input_folder", type=str, required=True,
                        help="Path to the input folder containing videos.")

    return parser.parse_args()

def main():
    model = "FunAudioLLM/SenseVoiceSmall"
    model = load_model(model)

    args = parse_args()
    base_path = os.path.abspath(args.input_folder)
    folder_path = os.path.join(base_path, "audios")
    output_json = os.path.join(base_path, "annotation(re).json")
    annotation_path = os.path.join(base_path, "annotation(new).json")
    frames_path = os.path.join(base_path, "frames")
    process_audio_folder(model, folder_path, output_json, annotation_path, frames_path)
    print(f"Emotion analysis results saved to {output_json}")

if __name__ == "__main__":
    main()
