"""AppraiseHate v11: richer CAT-guided implicit hate reasoning.

Design goals:
- Keep v10's separation between hateful content meaning and speaker stance.
- Produce a richer, denser T1 signal for implicit hate.
- Explicitly model mechanisms common in MultiHateClip:
  sexism/objectification, purity/shame, group comparison, stereotype, moral
  judgment, offensive humor/irony, exclusion, dehumanization, and threat.

The output remains compact enough for downstream text embedding.
"""

import argparse
import asyncio
import base64
import json
import logging
import os
import time
from datetime import datetime

from dotenv import load_dotenv
from openai import APIConnectionError, AsyncOpenAI, RateLimitError
from tqdm import tqdm

load_dotenv()
MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-4.1-nano")
client = AsyncOpenAI(
    base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
    api_key=os.getenv("OPENAI_API_KEY"),
)

QUAD_INTRO = (
    "You are analyzing a video represented by multiple 2x2 quad images. "
    "Each quad contains four consecutive frames in temporal order: top-left, "
    "top-right, bottom-left, bottom-right. The quads are in chronological "
    "order and represent the full video."
)


def setup_logger():
    os.makedirs("./logs", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"./logs/appraise_v11_{ts}.log"
    logger = logging.getLogger("V11")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fh = logging.FileHandler(log_file, encoding="utf-8")
    ch = logging.StreamHandler()
    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fh.setFormatter(fmt)
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.info(f"Log: {log_file}, Model: {MODEL_NAME}")
    return logger


def encode_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def build_image_content(frames):
    return [
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{encode_image(frame)}"},
        }
        for frame in frames
    ]


def _try_extract_json(text):
    text = text.strip()
    for prefix in ["```json", "```"]:
        if text.startswith(prefix):
            text = text[len(prefix):]
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass

    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    for i, char in enumerate(text[start:], start):
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
        if depth == 0:
            try:
                return json.loads(text[start : i + 1])
            except Exception:
                return None
    return None


async def request_with_retries(messages, max_tokens=1000, logger=None):
    for attempt in range(5):
        try:
            response = await client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.2,
            )
            return response.choices[0].message.content.strip()
        except (RateLimitError, APIConnectionError) as err:
            if logger:
                logger.warning(f"Retry {attempt + 1}/5: {type(err).__name__}")
            await asyncio.sleep(2**attempt)
        except Exception as err:
            if logger:
                logger.error(f"Error: {err}")
            if attempt < 4:
                await asyncio.sleep(2)
    return ""


async def run_v11_prompt(video_frames, title, transcript, emotion, logger=None):
    prompt = f"""{QUAD_INTRO}

Analyze whether this video CONTAINS or CONVEYS group-targeted hateful meaning.

Judge two axes separately:
1. CONTENT MEANING: what negative social message about a group is present?
2. SPEAKER STANCE: does the video endorse it, report/quote it, condemn it, mock it, or remain unclear?

Use Cognitive Appraisal Theory (CAT). Group-targeted hate often frames a target as:
- blameworthy
- threatening
- disgusting/contaminating
- inferior or dominated
- excludable
- less-than-human
- deserving punishment or harm

Internal analysis procedure:
1. Identify the strongest verbal, visual, and contextual cues.
2. Infer the social mechanism used, if any:
   stereotype, sexism/objectification, sexual shame/purity, group comparison,
   moral condemnation, offensive humor/irony, exclusion, dehumanization,
   threat framing, violent legitimization, or none.
3. State the implied group-level proposition, not just the literal words.
4. Map that proposition to CAT appraisals.
5. Decide speaker stance separately.

Important:
- Do NOT collapse content meaning into speaker stance.
- A video can REPORT, QUOTE, MOCK, or CRITICIZE hateful content while still CONTAINING that hateful message.
- Therefore, stance must NOT automatically force appraisal scores to zero.
- Use low scores only when no group-targeted hateful proposition is actually present.

Targeting rules:
- If an individual is attacked through a group-marked identity (e.g. woman, Muslim, Christian, gay, trans, Black), treat the broader group as targeted.
- Gendered sexual slurs such as "slut" or "whore" can target women when they rely on misogynistic norms rather than a purely personal insult.
- Group comparison can target one or both groups if superiority/inferiority is implied.
- Moral or religious judgment can count as implicit hate when a group is framed as sinful, impure, socially corrosive, inferior, or undeserving of equal standing.
- Offensive humor counts if the joke works by degrading, stereotyping, or normalizing contempt toward a group.
- Direct profanity with no group meaning should stay low.

Reportage protection:
- If hateful content is clearly quoted, contextualized, or criticized, set quoted_or_criticized=true.
- When quoted_or_criticized=true, appraisal scores may still be non-zero if the hateful proposition is clear and central in the content.
- News, education, and counter-speech should score near zero only when the hateful proposition is merely background and clearly rejected.

Scoring:
- 0 = absent
- 1 = implied / subtle / partial
- 2 = clear / central / repeated / strong

Video metadata:
- Title: {title or "N/A"}
- Transcript: {transcript or "N/A"}
- Voice emotion: {emotion or "N/A"}

Output requirements:
- Keep `surface_cues` short and concrete.
- `implicit_hate_mechanisms` must contain 1-3 items from this fixed list only:
  ["stereotype","sexism_objectification","sexual_shame_purity","group_comparison",
   "moral_condemnation","offensive_humor_irony","exclusion","dehumanization",
   "threat_framing","violent_legitimization","none"]
- `implicit_meaning` must be richer than a label: 2-3 short sentences, about 45-90 words, covering:
  (a) what stereotype/norm/judgment is activated,
  (b) how the target group is positioned socially,
  (c) whether the video endorses or instead reports/criticizes that meaning.
- For explicit hate on HateMM, say the direct hostile proposition plainly without unnecessary abstraction.
- Return JSON only with exactly these 8 top-level fields:

{{
  "target_group": "named social group, subgroup, or 'none'",
  "surface_cues": "1 short sentence listing the strongest visual/verbal/context cues.",
  "implicit_hate_mechanisms": ["none"],
  "appraisal_vector": {{
    "blame": 0,
    "threat": 0,
    "disgust": 0,
    "dominance": 0,
    "exclusion": 0,
    "dehumanization": 0,
    "harm_legitimization": 0
  }},
  "implicit_meaning": "2-3 short sentences (45-90 words) stating the implied group-level social message.",
  "contrastive_readings": {{
    "hateful": "Strongest hateful reading in <=25 words.",
    "non_hateful": "Strongest plausible non-hateful reading in <=25 words."
  }},
  "stance": "endorse / report / condemn / mock / unclear",
  "quoted_or_criticized": false
}}"""

    content = [{"type": "text", "text": prompt}] + build_image_content(video_frames)
    raw = ""
    for attempt in range(3):
        raw = await request_with_retries(
            [{"role": "user", "content": content}],
            max_tokens=800,
            logger=logger,
        )
        if not raw:
            continue
        parsed = _try_extract_json(raw)
        if parsed and "appraisal_vector" in parsed and "implicit_meaning" in parsed:
            return parsed, raw
        if logger:
            logger.warning(f"Parse fail {attempt + 1}/3")
    return (_try_extract_json(raw) if raw else {}), raw or ""


def is_valid(item):
    samples = item.get("v11_samples", [])
    return (
        len(samples) >= 1
        and "appraisal_vector" in samples[0]
        and "implicit_meaning" in samples[0]
    )


def load_data(data_path, save_path):
    with open(data_path, "r") as f:
        data = json.load(f)
    if os.path.exists(save_path):
        with open(save_path, "r") as f:
            saved = json.load(f)
        saved_map = {d.get("Video_ID"): d for d in saved if d.get("Video_ID")}
        for item in data:
            saved_item = saved_map.get(item.get("Video_ID"))
            if saved_item:
                item.update(saved_item)
    return data


async def process_item(
    item,
    data,
    save_path,
    quad_root,
    write_lock,
    semaphore,
    logger,
    n_samples=3,
):
    vid = item.get("Video_ID")
    existing = item.get("v11_samples", [])
    if len(existing) >= n_samples and is_valid(item):
        return "skipped"

    video_dir = os.path.join(quad_root, vid)
    if not os.path.isdir(video_dir):
        return "no_dir"

    frames = sorted(
        os.path.join(video_dir, name)
        for name in os.listdir(video_dir)
        if name.lower().endswith((".jpg", ".png"))
    )
    if not frames:
        return "no_frames"

    start = time.time()
    samples = existing.copy()
    async with semaphore:
        for _ in range(len(samples), n_samples):
            result, _ = await run_v11_prompt(
                frames,
                item.get("Title", ""),
                item.get("Transcript", ""),
                item.get("Emotion", ""),
                logger,
            )
            if result and "appraisal_vector" in result:
                samples.append(result)
    item["v11_samples"] = samples

    logger.info(
        f"{vid}: {'ok' if is_valid(item) else 'INCOMPLETE'} "
        f"({len(samples)} samples, {time.time() - start:.1f}s)"
    )
    async with write_lock:
        with open(save_path, "w") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
    return "ok" if is_valid(item) else "incomplete"


async def process(data_path, save_path, quad_root, max_concurrent, logger, n_samples=3):
    data = load_data(data_path, save_path)
    done = sum(1 for d in data if len(d.get("v11_samples", [])) >= n_samples)
    logger.info(f"Total: {len(data)}, Done: {done}, Need: {len(data) - done}")

    write_lock = asyncio.Lock()
    semaphore = asyncio.Semaphore(max_concurrent)
    stats = {"skipped": 0, "ok": 0, "incomplete": 0, "no_dir": 0, "no_frames": 0}
    pbar = tqdm(total=len(data), initial=done, desc="V11", unit="video")

    async def worker(item):
        try:
            result = await process_item(
                item,
                data,
                save_path,
                quad_root,
                write_lock,
                semaphore,
                logger,
                n_samples,
            )
            if result:
                stats[result] = stats.get(result, 0) + 1
        except Exception as err:
            logger.error(f"{item.get('Video_ID', '?')}: {err}")
        finally:
            pbar.update(1)

    await asyncio.gather(*[worker(item) for item in data])
    pbar.close()
    with open(save_path, "w") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    valid = sum(1 for d in data if is_valid(d))
    logger.info(f"Stats: {json.dumps(stats)}, Valid: {valid}/{len(data)}")


def get_dataset_paths(dataset_name, language="English"):
    if dataset_name == "HateMM":
        return (
            "./datasets/HateMM/annotation(re).json",
            "./datasets/HateMM/appraise_v11_data.json",
            "./datasets/HateMM/quad",
        )
    if dataset_name == "Multihateclip":
        return (
            f"./datasets/Multihateclip/{language}/annotation(new).json",
            f"./datasets/Multihateclip/{language}/appraise_v11_data.json",
            f"./datasets/Multihateclip/{language}/quad",
        )
    raise ValueError(f"Unknown dataset: {dataset_name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="HateMM",
        choices=["HateMM", "Multihateclip"],
    )
    parser.add_argument(
        "--language",
        type=str,
        default="English",
        choices=["English", "Chinese"],
    )
    parser.add_argument("--max_concurrent", type=int, default=10)
    parser.add_argument("--n_samples", type=int, default=3)
    args = parser.parse_args()

    logger = setup_logger()
    data_path, save_path, quad_root = get_dataset_paths(args.dataset_name, args.language)
    asyncio.run(
        process(
            data_path,
            save_path,
            quad_root,
            args.max_concurrent,
            logger,
            args.n_samples,
        )
    )
    logger.info("Done.")


if __name__ == "__main__":
    main()
