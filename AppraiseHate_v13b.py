"""AppraiseHate v13b: Perception-to-Cognition prompt for harmful video detection.

Single LLM call per video with structured 4-step reasoning (P2C-CoT).
Async concurrent execution following v12 pattern.
Model: gpt-5.4-nano
"""

import argparse, asyncio, base64, json, logging, os, re, time
from datetime import datetime
from dotenv import load_dotenv
from openai import APIConnectionError, AsyncOpenAI, RateLimitError
from tqdm import tqdm

load_dotenv()
MODEL_NAME = "gpt-5.4-nano"  # hardcoded, ignore env
client = AsyncOpenAI(
    base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
    api_key=os.getenv("OPENAI_API_KEY")
)

SYSTEM_PROMPT = ("You are a content moderation expert. You analyze videos to determine whether they contain hateful content. "
    "You are given a video represented by multiple 2x2 quad images. "
    "Each quad contains four consecutive frames in temporal order: top-left, top-right, bottom-left, bottom-right. "
    "The quads are in chronological order and represent the full video.")

USER_PROMPT_TEMPLATE = """Your task is to analyze whether the given video should be classified as "Harmful" or "Normal". Think before answering, and generate:
1. A structured reasoning process enclosed in <think></think> tags
2. A final explanation enclosed in <answer></answer> tags

A video should be classified as "Harmful" when it contains explicit or implicit abusive, insulting, degrading, humiliating, offensive, hateful, threatening, or hostile content toward a person, a social group, or a protected group. This includes severe harassment, slurs, demeaning stereotypes, dehumanization, contemptuous ridicule, exclusion, and incitement or endorsement of hostility. A video should be classified as "Normal" when it does not contain such harmful expression, or when potentially offensive language is only quoted, condemned, discussed, reported, or used in a clearly non-endorsing context.

Consider all available signals in the video, including visual scenes, on-screen text, subtitles, spoken language, gestures, symbols, and context.

For all videos, the reasoning should follow this structured 4-step process:
<think> must include exactly the following four steps enclosed in corresponding tags:
<step1>: Scene and Content Description — Provide an objective summary of the visible scene, participants, actions, on-screen text, and spoken or subtitle content.
<step2>: Harmful Cue Identification — Describe the specific abusive, offensive, hateful, insulting, threatening, degrading, or hostile cues in the video, if any. Mention where they appear when possible.
<step3>: Target and Attack Type Recognition — Identify who or what is being targeted, if anyone, and explain the type of attack involved, such as insult, harassment, humiliation, stereotype, dehumanization, exclusion, or threat. Distinguish endorsement from quotation, criticism, reporting, satire, or neutral discussion.
<step4>: Final Classification Reasoning — Explain why the content should be classified as "Harmful" or "Normal" overall, based on the severity, target, intent, and context.

Answer must be a single, coherent paragraph in natural language, which includes exactly the following six tags:
<which>: Define the video as "Harmful" or "Normal"
<what>: What harmful or non-harmful content appears in the video
<target>: Who is being targeted; if none, write "None"
<where>: Where the key evidence appears (e.g., subtitle area, center of frame, on-screen text, spoken by the main speaker)
<why>: Why the video is classified as Harmful or Normal
<how>: How the content could cause harm, reinforce hostility, normalize abuse, or, if Normal, why such harm is absent

Strict format requirement:
You must output exactly this structure, with every tag explicitly opened and closed, in the same order, and with no missing tags:
<think>
<step1>...</step1>
<step2>...</step2>
<step3>...</step3>
<step4>...</step4>
</think>
<answer>
<which>Harmful or Normal</which>
<what>...</what>
<target>...</target>
<where>...</where>
<why>...</why>
<how>...</how>
</answer>

Important instructions:
- Base the judgment on the video evidence and its context, not on assumptions.
- Classify as "Harmful" if the video clearly contains abusive, insulting, degrading, humiliating, offensive, hateful, or threatening expression, even if the target is an individual rather than a protected group.
- Do not require protected-group hate in order to classify a video as Harmful.
- However, do not classify content as Harmful when offensive language is only quoted, criticized, rebutted, reported, or discussed without endorsement.
- If the video contains severe insults, slurs, harassment, ridicule, contempt, or targeted hostility, do not downgrade it to "Normal" merely because the case is not full hate speech.
- Use "Normal" only when harmful intent or harmful expression is absent, weak, purely contextualized, or clearly non-endorsed.

Video Title: {title}
Video Transcript: {transcript}

Analyze the video frames shown above."""


def setup_logger():
    os.makedirs("./logs", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    lf = f"./logs/appraise_v13b_{ts}.log"
    logger = logging.getLogger("V13b"); logger.setLevel(logging.INFO); logger.handlers.clear()
    fh = logging.FileHandler(lf, encoding="utf-8"); ch = logging.StreamHandler()
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    fh.setFormatter(fmt); ch.setFormatter(fmt); logger.addHandler(fh); logger.addHandler(ch)
    logger.info(f"Log: {lf}, Model: {MODEL_NAME}"); return logger


def encode_image(p):
    with open(p, "rb") as f: return base64.b64encode(f.read()).decode("utf-8")


def build_image_content(frames):
    return [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encode_image(f)}"}} for f in frames]


def _strip_tag_value(value):
    cleaned = re.sub(r'^[\s:：\-–—]+', '', value.strip())
    cleaned = re.sub(r'(?:</[^>]+>\s*)+$', '', cleaned, flags=re.IGNORECASE)
    return cleaned.strip()


def _extract_block(text, tag, fallback_starts=()):
    start_match = re.search(rf'<{tag}>', text, re.IGNORECASE)
    if not start_match:
        return None
    start = start_match.end()

    end_candidates = []
    close_match = re.search(rf'</{tag}>', text[start:], re.IGNORECASE)
    if close_match:
        end_candidates.append(start + close_match.start())

    for fallback in fallback_starts:
        fallback_match = re.search(rf'<{fallback}>', text[start:], re.IGNORECASE)
        if fallback_match:
            end_candidates.append(start + fallback_match.start())

    end = min(end_candidates) if end_candidates else len(text)
    return text[start:end].strip()


def _extract_tag_value(text, tag, next_tags=(), block_end_tags=()):
    start_match = re.search(rf'(?:<{tag}>|</{tag}>)', text, re.IGNORECASE)
    if not start_match:
        return None
    start = start_match.end()

    end_candidates = []
    close_match = re.search(rf'</{tag}>', text[start:], re.IGNORECASE)
    if close_match:
        end_candidates.append(start + close_match.start())

    for next_tag in next_tags:
        for boundary in (rf'<{next_tag}>', rf'</{next_tag}>'):
            next_match = re.search(boundary, text[start:], re.IGNORECASE)
            if next_match:
                end_candidates.append(start + next_match.start())

    for end_tag in block_end_tags:
        end_match = re.search(rf'</{end_tag}>', text[start:], re.IGNORECASE)
        if end_match:
            end_candidates.append(start + end_match.start())

    end = min(end_candidates) if end_candidates else len(text)
    value = _strip_tag_value(text[start:end])
    return value or None


def _normalize_which_label(value):
    if not value:
        return None
    lowered = value.lower()
    if 'normal' in lowered:
        return 'Normal'
    if 'harmful' in lowered:
        return 'Harmful'
    return _strip_tag_value(value)


def parse_response(text):
    """Parse partially malformed tagged responses for Harmful/Normal outputs."""
    result = {}

    think_text = _extract_block(text, 'think', fallback_starts=('answer',))
    if think_text:
        result['think'] = think_text
        step_tags = [f'step{i}' for i in range(1, 5)]
        for idx, step_tag in enumerate(step_tags):
            value = _extract_tag_value(
                think_text,
                step_tag,
                next_tags=step_tags[idx + 1:],
                block_end_tags=('think', 'answer')
            )
            if value:
                result[step_tag] = value

    answer_text = _extract_block(text, 'answer')
    search_text = answer_text if answer_text else text
    if answer_text:
        result['answer'] = answer_text

    answer_tags = ['which', 'what', 'target', 'where', 'why', 'how']
    for idx, tag in enumerate(answer_tags):
        value = _extract_tag_value(
            search_text,
            tag,
            next_tags=answer_tags[idx + 1:],
            block_end_tags=('answer',)
        )
        if value:
            result[tag] = value

    result['which'] = _normalize_which_label(result.get('which'))

    return {k: v for k, v in result.items() if v is not None}


async def request_with_retries(messages, max_tokens=2048, logger=None):
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
    resp = item.get("v13_response", {})
    return resp.get("step1") and resp.get("which")


def load_data(data_path, save_path):
    with open(data_path, "r") as f: data = json.load(f)
    if os.path.exists(save_path):
        with open(save_path, "r") as f: saved = json.load(f)
        sm = {d.get("Video_ID"): d for d in saved if d.get("Video_ID")}
        for item in data:
            s = sm.get(item.get("Video_ID"))
            if s and "v13_response" in s: item["v13_response"] = s["v13_response"]
    return data


def get_dataset_paths(dataset_name, language="English"):
    if dataset_name == "HateMM":
        return ("./datasets/HateMM/annotation(new).json",
                "./datasets/HateMM/appraise_v13b_data.json",
                "./datasets/HateMM/quad")
    elif dataset_name == "ImpliHateVid":
        return ("./datasets/ImpliHateVid/annotation(new).json",
                "./datasets/ImpliHateVid/appraise_v13b_data.json",
                "./datasets/ImpliHateVid/quad")
    else:
        return (f"./datasets/Multihateclip/{language}/annotation(new).json",
                f"./datasets/Multihateclip/{language}/appraise_v13b_data.json",
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
            max_tokens=2048, logger=logger
        )

    if raw:
        parsed = parse_response(raw)
        parsed['raw'] = raw
        item['v13_response'] = parsed
    else:
        item['v13_response'] = {'error': 'empty response', 'raw': ''}

    which = item['v13_response'].get('which', 'unknown')
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
    pbar = tqdm(total=len(data), initial=done, desc="V13", unit="video")

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
    parser.add_argument("--dataset_name", type=str, default="HateMM", choices=["HateMM", "Multihateclip", "ImpliHateVid"])
    parser.add_argument("--language", type=str, default="English", choices=["English", "Chinese"])
    parser.add_argument("--max_concurrent", type=int, default=10)
    args = parser.parse_args()

    logger = setup_logger()
    data_path, save_path, quad_root = get_dataset_paths(args.dataset_name, args.language)
    asyncio.run(process(data_path, save_path, quad_root, args.max_concurrent, logger))
    logger.info("Done.")


if __name__ == "__main__":
    main()
