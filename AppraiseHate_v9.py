"""AppraiseHate v9: v5 + minimal stance addition.

Exact v5 prompt + ONE extra field: stance (endorse/report/condemn/mock/unclear).
Tests whether stance alone improves v5 without hurting its proven text quality.

If v9 > v5: stance helps → Step 1 improved, move to Step 2
If v9 = v5: stance is neutral → keep v5, move to Step 2
If v9 < v5: adding stance hurts → stick with v5

Multi-sample (N=3).
"""

import argparse, asyncio, base64, json, logging, os, time
from datetime import datetime
from dotenv import load_dotenv
from openai import APIConnectionError, AsyncOpenAI, RateLimitError
from tqdm import tqdm

load_dotenv()
MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-4.1-nano")
client = AsyncOpenAI(base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"), api_key=os.getenv("OPENAI_API_KEY"))

QUAD_INTRO = ("You are analyzing a video represented by multiple 2x2 quad images. "
    "Each quad contains four consecutive frames in temporal order: top-left, top-right, bottom-left, bottom-right. "
    "The quads are in chronological order and represent the full video.")

def setup_logger():
    os.makedirs("./logs", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    lf = f"./logs/appraise_v9_{ts}.log"
    logger = logging.getLogger("V9"); logger.setLevel(logging.INFO); logger.handlers.clear()
    fh = logging.FileHandler(lf, encoding="utf-8"); ch = logging.StreamHandler()
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    fh.setFormatter(fmt); ch.setFormatter(fmt); logger.addHandler(fh); logger.addHandler(ch)
    logger.info(f"Log: {lf}, Model: {MODEL_NAME}"); return logger

def encode_image(p):
    with open(p, "rb") as f: return base64.b64encode(f.read()).decode("utf-8")
def build_image_content(frames):
    return [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encode_image(f)}"}} for f in frames]
def _try_extract_json(text):
    text = text.strip()
    for pfx in ["```json", "```"]:
        if text.startswith(pfx): text = text[len(pfx):]; text = text[:-3] if text.endswith("```") else text; text = text.strip()
    try: return json.loads(text)
    except: pass
    start = text.find("{")
    if start < 0: return None
    depth = 0
    for i, c in enumerate(text[start:], start):
        if c == "{": depth += 1
        elif c == "}": depth -= 1
        if depth == 0:
            try: return json.loads(text[start:i+1])
            except: return None
    return None

async def request_with_retries(messages, max_tokens=800, logger=None):
    for attempt in range(5):
        try:
            r = await client.chat.completions.create(model=MODEL_NAME, messages=messages, max_tokens=max_tokens, temperature=0.3)
            return r.choices[0].message.content.strip()
        except (RateLimitError, APIConnectionError) as e:
            if logger: logger.warning(f"Retry {attempt+1}/5: {type(e).__name__}")
            await asyncio.sleep(2 ** attempt)
        except Exception as e:
            if logger: logger.error(f"Error: {e}")
            if attempt < 4: await asyncio.sleep(2)
    return ""

async def run_v9_prompt(video_frames, title, transcript, emotion, logger=None):
    prompt = f"""{QUAD_INTRO}

Analyze whether this video conveys hate toward a social group.

Use cognitive appraisal theory: hateful content often frames a target as blameworthy, threatening, contaminating/disgusting, inferior/dominated, excludable, less-than-human, or deserving harm.

Video metadata:
- Title: {title or "N/A"}
- Transcript: {transcript or "N/A"}
- Voice emotion: {emotion or "N/A"}

Instructions:
- Consider visuals, speech, on-screen text, tone, and context jointly.
- Focus on implied meaning, not just literal wording.
- Score each appraisal from 0 (absent) to 2 (clear/strong).
- A video can SHOW hate without ENDORSING it. News reporting, counter-speech, education, and criticism should not be classified as hateful.
- Keep outputs short.
- Return JSON only with exactly these 4 top-level fields:

{{
  "appraisal_vector": {{
    "blame": 0,
    "threat": 0,
    "disgust": 0,
    "dominance": 0,
    "exclusion": 0,
    "dehumanization": 0,
    "harm_legitimization": 0
  }},
  "implicit_meaning": "1-2 sentences stating the implied social message beyond literal content, naming the target if present.",
  "contrastive_readings": {{
    "hateful": "Strongest hateful reading in <=25 words.",
    "non_hateful": "Strongest plausible non-hateful reading in <=25 words."
  }},
  "stance": "endorse / report / condemn / mock / unclear"
}}"""

    content = [{"type": "text", "text": prompt}] + build_image_content(video_frames)
    for attempt in range(3):
        raw = await request_with_retries([{"role": "user", "content": content}], max_tokens=600, logger=logger)
        if not raw: continue
        parsed = _try_extract_json(raw)
        if parsed and "appraisal_vector" in parsed and "implicit_meaning" in parsed:
            return parsed, raw
        if logger: logger.warning(f"Parse fail {attempt+1}/3")
    return _try_extract_json(raw) if raw else {}, raw or ""

def is_valid(item):
    samples = item.get("v9_samples", [])
    return len(samples) >= 1 and "appraisal_vector" in samples[0] and "implicit_meaning" in samples[0]

def load_data(data_path, save_path):
    with open(data_path, "r") as f: data = json.load(f)
    if os.path.exists(save_path):
        with open(save_path, "r") as f: saved = json.load(f)
        sm = {d.get("Video_ID"): d for d in saved if d.get("Video_ID")}
        for item in data:
            s = sm.get(item.get("Video_ID"))
            if s: item.update(s)
    return data

async def process_item(item, data, save_path, quad_root, write_lock, semaphore, logger, n_samples=3):
    vid = item.get("Video_ID")
    existing = item.get("v9_samples", [])
    if len(existing) >= n_samples and is_valid(item): return "skipped"
    vp = os.path.join(quad_root, vid)
    if not os.path.isdir(vp): return "no_dir"
    frames = sorted([os.path.join(vp, f) for f in os.listdir(vp) if f.lower().endswith((".jpg", ".png"))])
    if not frames: return "no_frames"
    t0 = time.time(); samples = existing.copy()
    async with semaphore:
        for i in range(len(samples), n_samples):
            result, raw = await run_v9_prompt(frames, item.get("Title",""), item.get("Transcript",""), item.get("Emotion",""), logger)
            if result and "appraisal_vector" in result: samples.append(result)
    item["v9_samples"] = samples
    logger.info(f"{vid}: {'ok' if is_valid(item) else 'INCOMPLETE'} ({len(samples)} samples, {time.time()-t0:.1f}s)")
    async with write_lock:
        with open(save_path, "w") as f: json.dump(data, f, ensure_ascii=False, indent=4)
    return "ok" if is_valid(item) else "incomplete"

async def process(data_path, save_path, quad_root, max_concurrent, logger, n_samples=3):
    data = load_data(data_path, save_path)
    done = sum(1 for d in data if len(d.get("v9_samples",[])) >= n_samples)
    logger.info(f"Total: {len(data)}, Done: {done}, Need: {len(data)-done}")
    write_lock = asyncio.Lock(); semaphore = asyncio.Semaphore(max_concurrent)
    stats = {"skipped":0,"ok":0,"incomplete":0,"no_dir":0,"no_frames":0}
    pbar = tqdm(total=len(data), initial=done, desc="V9", unit="video")
    async def w(item):
        try:
            r = await process_item(item, data, save_path, quad_root, write_lock, semaphore, logger, n_samples)
            if r: stats[r] = stats.get(r,0)+1
        except Exception as e: logger.error(f"{item.get('Video_ID','?')}: {e}")
        finally: pbar.update(1)
    await asyncio.gather(*[w(item) for item in data])
    pbar.close()
    with open(save_path, "w") as f: json.dump(data, f, ensure_ascii=False, indent=4)
    valid = sum(1 for d in data if is_valid(d))
    logger.info(f"Stats: {json.dumps(stats)}, Valid: {valid}/{len(data)}")

def get_dataset_paths(dataset_name, language="English"):
    if dataset_name == "HateMM":
        return ("./datasets/HateMM/annotation(re).json",
                "./datasets/HateMM/appraise_v9_data.json",
                "./datasets/HateMM/quad")
    elif dataset_name == "Multihateclip":
        return (f"./datasets/Multihateclip/{language}/annotation(new).json",
                f"./datasets/Multihateclip/{language}/appraise_v9_data.json",
                f"./datasets/Multihateclip/{language}/quad")
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="HateMM", choices=["HateMM", "Multihateclip"])
    parser.add_argument("--language", type=str, default="English", choices=["English", "Chinese"])
    parser.add_argument("--max_concurrent", type=int, default=10)
    parser.add_argument("--n_samples", type=int, default=3)
    args = parser.parse_args()
    logger = setup_logger()
    data_path, save_path, quad_root = get_dataset_paths(args.dataset_name, args.language)
    asyncio.run(process(data_path, save_path, quad_root, args.max_concurrent, logger, args.n_samples))
    logger.info("Done.")

if __name__ == "__main__":
    main()
