"""AppraiseHate v12: Grounded Evidence Ledger + CAT Judge (2-call pipeline).

Call 1: Extract factual evidence from video (grounding) — sees quad images
Call 2: Apply CAT-guided reasoning on evidence only — pure text, no images

Inspired by:
- ARGUS (CVPR 2025): separate grounding from reasoning
- STEP (CVPR 2025): structured decomposition for video understanding
- Third-Person Appraisal Agent (EMNLP 2025 Findings): CAT as reasoning scaffold
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
    lf = f"./logs/appraise_v12_{ts}.log"
    logger = logging.getLogger("V12"); logger.setLevel(logging.INFO); logger.handlers.clear()
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

# ============================================================
# Call 1: Evidence Ledger (sees images)
# ============================================================
async def call1_evidence(video_frames, title, transcript, emotion, logger=None):
    prompt = f"""{QUAD_INTRO}

Extract a compact evidence ledger from this video. Focus on FACTS, not judgment.

Video metadata:
- Title: {title or "N/A"}
- Transcript: {transcript or "N/A"}
- Voice emotion: {emotion or "N/A"}

Extract and return JSON:
{{
  "visual_content": "What do the video frames show? Describe key people, actions, symbols, text overlays, settings. 2-3 sentences.",
  "spoken_content": "What is said in the transcript? Summarize the key message, noting any slurs, insults, stereotypes, or coded language. 1-2 sentences.",
  "tone_and_framing": "What is the overall tone? Is it serious, humorous, satirical, educational, angry, mocking? How is the content framed — as endorsement, criticism, reporting, entertainment? 1 sentence.",
  "target_group": "Which social group (if any) is discussed, depicted, or addressed?",
  "key_cues": ["list", "of", "3-5", "most", "important", "evidence", "phrases"]
}}

Rules:
- Be factual and specific. Quote actual words/phrases when possible.
- Do NOT judge whether content is hateful yet.
- Return JSON only."""

    content = [{"type": "text", "text": prompt}] + build_image_content(video_frames)
    for attempt in range(3):
        raw = await request_with_retries([{"role": "user", "content": content}], max_tokens=600, logger=logger)
        if not raw: continue
        parsed = _try_extract_json(raw)
        if parsed and parsed.get("visual_content"):
            return parsed, raw
        if logger: logger.warning(f"Call1 parse fail {attempt+1}/3")
    return _try_extract_json(raw) if raw else {}, raw or ""

# ============================================================
# Call 2: CAT Judge (text only, no images)
# ============================================================
async def call2_judge(evidence, logger=None):
    evidence_text = json.dumps(evidence, ensure_ascii=False, indent=2)
    prompt = f"""You are a content analyst. Based on the evidence ledger below, determine whether this video conveys hateful or offensive content toward a social group.

Evidence ledger:
{evidence_text}

Use Cognitive Appraisal Theory as your reasoning guide. Internally check:
- Is the target group framed as BLAMEWORTHY for problems?
- Is the target framed as THREATENING or dangerous?
- Are there DISGUST or contamination cues toward the target?
- Is the target framed as INFERIOR or dominated?
- Is EXCLUSION of the target justified or normalized?
- Is there DEHUMANIZATION (animal metaphors, objectification)?
- Is HARM toward the target legitimized?

Then critically assess:
- Does the VIDEO CREATOR endorse this framing, or merely report/quote/satirize/condemn it?
- Satire that mocks hateful ideas is NOT hateful.
- News reporting or education that shows hateful content is NOT hateful.
- Counter-speech that condemns hate is NOT hateful.
- But content that uses humor/irony to SPREAD stereotypes or degrade groups IS hateful.

Return JSON:
{{
  "implicit_meaning": "What is the video REALLY implying about the target group? Consider the tone, framing, and creator intent. Be thorough.",
  "contrastive_readings": {{
    "hateful": "The strongest hateful interpretation of this video.",
    "non_hateful": "The strongest non-hateful interpretation of this video."
  }},
  "speaker_stance": "endorse / report / condemn / mock / unclear",
  "target_group": "the specific group targeted, or none"
}}

Rules:
- Focus on IMPLIED meaning, not literal words.
- Be specific about WHY you chose the stance.
- Return JSON only."""

    for attempt in range(3):
        raw = await request_with_retries([{"role": "user", "content": [{"type": "text", "text": prompt}]}], max_tokens=1000, logger=logger)
        if not raw: continue
        parsed = _try_extract_json(raw)
        if parsed and parsed.get("implicit_meaning"):
            return parsed, raw
        if logger: logger.warning(f"Call2 parse fail {attempt+1}/3")
    return _try_extract_json(raw) if raw else {}, raw or ""

# ============================================================
# Pipeline
# ============================================================
def is_valid(item):
    samples = item.get("v12_samples", [])
    return len(samples) >= 1 and "implicit_meaning" in samples[0]

def load_data(data_path, save_path):
    with open(data_path, "r") as f: data = json.load(f)
    if os.path.exists(save_path):
        with open(save_path, "r") as f: saved = json.load(f)
        sm = {d.get("Video_ID"): d for d in saved if d.get("Video_ID")}
        for item in data:
            s = sm.get(item.get("Video_ID"))
            if s: item.update(s)
    return data

def get_dataset_paths(dataset_name, language="English"):
    if dataset_name == "HateMM":
        return ("./datasets/HateMM/annotation(re).json",
                "./datasets/HateMM/appraise_v12_data.json",
                "./datasets/HateMM/quad")
    else:
        return (f"./datasets/Multihateclip/{language}/annotation(new).json",
                f"./datasets/Multihateclip/{language}/appraise_v12_data.json",
                f"./datasets/Multihateclip/{language}/quad")

async def process_item(item, data, save_path, quad_root, write_lock, semaphore, logger, n_samples=3):
    vid = item.get("Video_ID")
    existing = item.get("v12_samples", [])
    if len(existing) >= n_samples and is_valid(item): return "skipped"
    vp = os.path.join(quad_root, vid)
    if not os.path.isdir(vp): return "no_dir"
    frames = sorted([os.path.join(vp, f) for f in os.listdir(vp) if f.lower().endswith((".jpg", ".png"))])
    if not frames: return "no_frames"
    t0 = time.time(); samples = existing.copy()
    async with semaphore:
        for i in range(len(samples), n_samples):
            # Call 1: Evidence extraction (sees images)
            evidence, _ = await call1_evidence(frames, item.get("Title",""), item.get("Transcript",""), item.get("Emotion",""), logger)
            if not evidence.get("visual_content"):
                continue
            # Call 2: CAT judgment (text only)
            judgment, _ = await call2_judge(evidence, logger)
            if judgment and judgment.get("implicit_meaning"):
                judgment["evidence"] = evidence
                samples.append(judgment)
    item["v12_samples"] = samples
    logger.info(f"{vid}: {'ok' if is_valid(item) else 'INCOMPLETE'} ({len(samples)} samples, {time.time()-t0:.1f}s)")
    async with write_lock:
        with open(save_path, "w") as f: json.dump(data, f, ensure_ascii=False, indent=4)
    return "ok" if is_valid(item) else "incomplete"

async def process(data_path, save_path, quad_root, max_concurrent, logger, n_samples=3):
    data = load_data(data_path, save_path)
    done = sum(1 for d in data if len(d.get("v12_samples",[])) >= n_samples)
    logger.info(f"Total: {len(data)}, Done: {done}, Need: {len(data)-done}")
    write_lock = asyncio.Lock(); semaphore = asyncio.Semaphore(max_concurrent)
    stats = {"skipped":0,"ok":0,"incomplete":0,"no_dir":0,"no_frames":0}
    pbar = tqdm(total=len(data), initial=done, desc="V12", unit="video")
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
