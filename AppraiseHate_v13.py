"""AppraiseHate v13: Perception-to-Cognition prompt for hateful video detection.

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

USER_PROMPT_TEMPLATE = """Your task is to analyze whether the given video contains hateful content or not. Think before answering, and generate:
1. A structured reasoning process enclosed in <think></think> tags
2. A final explanation enclosed in <answer></answer> tags

A video should be classified as "Hateful" only when the content explicitly or implicitly expresses hatred, demeaning stereotypes, dehumanization, exclusion, hostility, or incitement toward a protected group or a clearly targeted social group. Consider all available signals in the video, including visual scenes, on-screen text, subtitles, spoken language, gestures, symbols, and their context. Distinguish hateful endorsement from neutral quotation, condemnation, reporting, satire, or discussion.

For hateful videos, the reasoning should be based on a structured 4-step process:
<think> must include the following four steps enclosed in corresponding tags:
<step1>: Scene and Context Description — Provide an objective overview of the environment, participants, visible text, symbols, and overall activity, without making any judgment yet.
<step2>: Hateful Evidence Description — Describe the specific verbal, textual, visual, or symbolic content that may indicate hateful expression, including its approximate spatial location when possible, without yet concluding why it is hateful.
<step3>: Target and Intent Recognition — Identify the likely target of the expression and explain whether the content appears to insult, stereotype, dehumanize, threaten, exclude, or encourage hostility toward that target. Also assess whether the video is endorsing the hateful message or merely quoting, criticizing, documenting, or discussing it.
<step4>: Contextual Harm and Norm Reasoning — Explain why the content should be considered hateful in context, including its potential social harm, the kind of hostility it promotes, and why the message goes beyond ordinary disagreement or non-hateful offensive language.

Answer must be a single, coherent paragraph in natural language, which includes exactly the following six tags:
<which>: Define the video as "Hateful"
<what>: What hateful content appears in the video
<target>: Who is being targeted
<where>: Where the key evidence appears (e.g., subtitle area, center of frame, on a poster, spoken by the main speaker)
<why>: Why the content is classified as hateful
<how>: How the content could cause harm, encourage hostility, or reinforce harmful prejudice

Strict format requirement for hateful videos:
You must output exactly this structure, with every tag explicitly opened and closed, in the same order, and with no missing closing tags:
<think>
<step1>...</step1>
<step2>...</step2>
<step3>...</step3>
<step4>...</step4>
</think>
<answer>
<which>Hateful</which>
<what>...</what>
<target>...</target>
<where>...</where>
<why>...</why>
<how>...</how>
</answer>

For non-hateful videos, the reasoning should be simplified to just two steps:
<think> must include only the following two steps:
<step1>: Scene and Content Description — Provide a concise and objective overview of the environment, people, text, speech, and actions in the video.
<step2>: Non-Hateful Explanation — Explain why the content is not hateful, including whether potentially sensitive language is absent, merely quoted, contextually condemned, satirical without endorsing hate, or otherwise not expressing hostility toward a target group.

Answer must be a single, coherent paragraph in natural language, which includes the following four tags:
<which>: Define the video as "Non-hateful"
<what>: A concise description of the video content
<target>: The apparent target, if any; otherwise "None"
<why>: Why it is not classified as hateful

Strict format requirement for non-hateful videos:
You must output exactly this structure, with every tag explicitly opened and closed, in the same order, and with no extra step tags:
<think>
<step1>...</step1>
<step2>...</step2>
</think>
<answer>
<which>Non-hateful</which>
<what>...</what>
<target>...</target>
<why>...</why>
</answer>

Important instructions:
- Base the judgment on the video evidence and its context, not on assumptions.
- If hateful cues are ambiguous, weak, or only indirectly implied, be cautious and explain the uncertainty inside the reasoning.
- Do not classify content as hateful solely because it is offensive, rude, politically charged, or controversial; it must involve hostility, degrading characterization, exclusion, or incitement toward a target group or clearly targeted social group.
- Pay special attention to whether the hateful content is endorsed by the video or presented for criticism, reporting, education, or rebuttal.

Video Title: {title}
Video Transcript: {transcript}

Analyze the video frames shown above."""


def setup_logger():
    os.makedirs("./logs", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    lf = f"./logs/appraise_v13_{ts}.log"
    logger = logging.getLogger("V13"); logger.setLevel(logging.INFO); logger.handlers.clear()
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
    if 'non-hateful' in lowered or 'non hateful' in lowered:
        return 'Non-hateful'
    if 'hateful' in lowered:
        return 'Hateful'
    return _strip_tag_value(value)


def parse_response(text):
    """Parse partially malformed tagged responses without defaulting hateful items to non-hateful."""
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
    is_hateful = result.get('which') == 'Hateful'
    is_non_hateful = result.get('which') == 'Non-hateful'

    if is_non_hateful:
        result.setdefault('where', 'None, the video does not contain hateful content')
        result.setdefault('how', 'None, the video does not promote harm or hostility')
        result.setdefault('step3', 'Not applicable, the video is non-hateful')
        result.setdefault('step4', 'Not applicable, the video is non-hateful')
    elif is_hateful:
        result.pop('step3', None) if result.get('step3', '').startswith('Not applicable') else None
        result.pop('step4', None) if result.get('step4', '').startswith('Not applicable') else None

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
                "./datasets/HateMM/appraise_v13_data.json",
                "./datasets/HateMM/quad")
    else:
        return (f"./datasets/Multihateclip/{language}/annotation(new).json",
                f"./datasets/Multihateclip/{language}/appraise_v13_data.json",
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
