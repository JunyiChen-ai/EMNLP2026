"""Baseline: Direct LLM classification without any structured reasoning.

Just asks the LLM to classify the video as Hateful or Non-hateful.
Same model (gpt-5.4-nano), same images, same title/transcript.
"""

import argparse, asyncio, base64, json, logging, os, re, time
from datetime import datetime
from dotenv import load_dotenv
from openai import APIConnectionError, AsyncOpenAI, RateLimitError
from tqdm import tqdm

load_dotenv()
MODEL_NAME = "gpt-5.4-nano"
client = AsyncOpenAI(
    base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
    api_key=os.getenv("OPENAI_API_KEY")
)

SYSTEM_PROMPT = "You are a content moderation expert."

USER_PROMPT_TEMPLATE = """Please classify the following video as either "Hateful" or "Non-hateful".

A video is "Hateful" if it expresses hatred, hostility, dehumanization, or incitement toward a target group. A video is "Non-hateful" if it does not.

Video Title: {title}
Video Transcript: {transcript}

Analyze the video frames shown above and respond with exactly one word: Hateful or Non-hateful."""


def setup_logger():
    os.makedirs("./logs", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    lf = f"./logs/baseline_direct_{ts}.log"
    logger = logging.getLogger("baseline"); logger.setLevel(logging.INFO); logger.handlers.clear()
    fh = logging.FileHandler(lf, encoding="utf-8"); ch = logging.StreamHandler()
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    fh.setFormatter(fmt); ch.setFormatter(fmt); logger.addHandler(fh); logger.addHandler(ch)
    logger.info(f"Log: {lf}, Model: {MODEL_NAME}"); return logger


def encode_image(p):
    with open(p, "rb") as f: return base64.b64encode(f.read()).decode("utf-8")


def build_image_content(frames):
    return [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encode_image(f)}"}} for f in frames]


def parse_response(text):
    lowered = text.strip().lower()
    if 'non-hateful' in lowered or 'non hateful' in lowered:
        return 'Non-hateful'
    if 'hateful' in lowered:
        return 'Hateful'
    return text.strip()


async def request_with_retries(messages, max_tokens=64, logger=None):
    for attempt in range(5):
        try:
            r = await client.chat.completions.create(
                model=MODEL_NAME, messages=messages, max_completion_tokens=max_tokens, temperature=0
            )
            return r.choices[0].message.content.strip()
        except (RateLimitError, APIConnectionError) as e:
            if logger: logger.warning(f"Retry {attempt+1}/5: {type(e).__name__}")
            await asyncio.sleep(2 ** attempt)
        except Exception as e:
            if logger: logger.error(f"Error: {e}")
            if attempt < 4: await asyncio.sleep(2)
    return ""


def is_valid(item):
    return item.get("baseline_response", {}).get("which") in ("Hateful", "Non-hateful")


def load_data(data_path, save_path):
    with open(data_path, "r") as f: data = json.load(f)
    if os.path.exists(save_path):
        with open(save_path, "r") as f: saved = json.load(f)
        sm = {d.get("Video_ID"): d for d in saved if d.get("Video_ID")}
        for item in data:
            s = sm.get(item.get("Video_ID"))
            if s and "baseline_response" in s: item["baseline_response"] = s["baseline_response"]
    return data


def get_dataset_paths(dataset_name, language="English"):
    if dataset_name == "HateMM":
        return ("./datasets/HateMM/annotation(new).json",
                "./datasets/HateMM/baseline_direct_data.json",
                "./datasets/HateMM/quad")
    else:
        return (f"./datasets/Multihateclip/{language}/annotation(new).json",
                f"./datasets/Multihateclip/{language}/baseline_direct_data.json",
                f"./datasets/Multihateclip/{language}/quad")


async def process_item(item, data, save_path, quad_root, write_lock, semaphore, logger):
    vid = item.get("Video_ID")
    if is_valid(item): return "skipped"

    vp = os.path.join(quad_root, vid)
    if not os.path.isdir(vp): return "no_dir"
    frames = sorted([os.path.join(vp, f) for f in os.listdir(vp) if f.lower().endswith((".jpg", ".png"))])
    if not frames: return "no_frames"

    title = item.get("Title", "") or ""
    transcript = item.get("Transcript", "") or ""
    prompt = USER_PROMPT_TEMPLATE.format(title=title, transcript=transcript)

    t0 = time.time()
    async with semaphore:
        content = build_image_content(frames) + [{"type": "text", "text": prompt}]
        raw = await request_with_retries(
            [{"role": "system", "content": SYSTEM_PROMPT},
             {"role": "user", "content": content}],
            max_tokens=64, logger=logger
        )

    which = parse_response(raw) if raw else "error"
    item['baseline_response'] = {'which': which, 'raw': raw}
    logger.info(f"{vid}: {which} ({time.time()-t0:.1f}s)")

    async with write_lock:
        with open(save_path, "w") as f: json.dump(data, f, ensure_ascii=False, indent=2)

    return "ok" if is_valid(item) else "incomplete"


async def process(data_path, save_path, quad_root, max_concurrent, logger):
    data = load_data(data_path, save_path)
    done = sum(1 for d in data if is_valid(d))
    logger.info(f"Total: {len(data)}, Done: {done}, Need: {len(data)-done}")

    write_lock = asyncio.Lock()
    semaphore = asyncio.Semaphore(max_concurrent)
    stats = {"skipped": 0, "ok": 0, "incomplete": 0, "no_dir": 0, "no_frames": 0}
    pbar = tqdm(total=len(data), initial=done, desc="Baseline", unit="video")

    async def w(item):
        try:
            r = await process_item(item, data, save_path, quad_root, write_lock, semaphore, logger)
            if r: stats[r] = stats.get(r, 0) + 1
        except Exception as e:
            logger.error(f"{item.get('Video_ID', '?')}: {e}")
        finally:
            pbar.update(1)

    await asyncio.gather(*[w(item) for item in data])
    pbar.close()

    with open(save_path, "w") as f: json.dump(data, f, ensure_ascii=False, indent=2)
    valid = sum(1 for d in data if is_valid(d))
    logger.info(f"Stats: {json.dumps(stats)}, Valid: {valid}/{len(data)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="HateMM", choices=["HateMM", "Multihateclip"])
    parser.add_argument("--language", type=str, default="English", choices=["English", "Chinese"])
    parser.add_argument("--max_concurrent", type=int, default=10)
    args = parser.parse_args()

    logger = setup_logger()
    data_path, save_path, quad_root = get_dataset_paths(args.dataset_name, args.language)
    asyncio.run(process(data_path, save_path, quad_root, args.max_concurrent, logger))
    logger.info("Done.")


if __name__ == "__main__":
    main()
