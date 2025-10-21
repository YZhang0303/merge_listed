import asyncio
import argparse
import json
import re
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import pandas as pd
from json_repair import repair_json
from openai import AsyncOpenAI
from openai import APIConnectionError, RateLimitError, APIStatusError
from tqdm import tqdm


def load_prompt(prompt_path: Path) -> str:
    with prompt_path.open("r", encoding="utf-8") as f:
        return f.read().strip()


def read_csv(input_path: Path) -> pd.DataFrame:
    # Use utf-8-sig to gracefully handle BOM if present
    df = pd.read_csv(input_path, encoding="utf-8-sig")
    expected_cols = {"Original_DWA_Title"}
    missing = expected_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"CSV is missing required column 'Original_DWA_Title'. Columns found: {list(df.columns)}"
        )
    return df


def load_processed_indices(output_jsonl: Path) -> Set[int]:
    processed: Set[int] = set()
    if not output_jsonl.exists():
        return processed
    with output_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                idx = int(obj.get("index"))
                processed.add(idx)
            except Exception:
                # skip malformed lines
                continue
    return processed


def normalize_json_content(content: str) -> Dict[str, Any]:
    """Try to turn model content into a JSON object with at least 'label'."""
    text = content.strip()
    # Strip markdown code fences if present
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9_+-]*\n", "", text)
        if text.endswith("```"):
            text = text[:-3]

    # Try strict JSON first
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # Try repairing JSON
    try:
        repaired = repair_json(text)
        obj = json.loads(repaired)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # Fallback: extract E0/E1/E2/E3 from free text
    match = re.search(r"\b(E[0-3])\b", text)
    if match:
        return {"label": match.group(1), "raw": text}

    raise ValueError("Model output is not valid JSON and label could not be inferred.")


def build_input(system_prompt: str, english: str) -> str:
    instruction = (
        "You are given one DWA task description in English. "
        "Classify it into exactly one label among E0, E1, E2, E3 per the taxonomy. "
        "Return strict JSON with keys: label (one of E0/E1/E2/E3) and explanation (<= 40 words)."
    )
    return (
        f"SYSTEM\n{system_prompt}\n\n"
        f"TASK\n{instruction}\n\n"
        f"INPUT\nEnglish: {english}\n\n"
        f"CONSTRAINTS\nRespond with JSON only."
    )


async def call_with_retries(
    client: AsyncOpenAI,
    model: str,
    input_text: str,
    effort: str,
    max_retries: int = 6,
    initial_backoff_s: float = 1.0,
) -> Dict[str, Any]:
    attempt = 0
    backoff = initial_backoff_s
    while True:
        attempt += 1
        try:
            resp = await client.responses.create(
                model=model,
                input=input_text,
                temperature=0,
                reasoning={"effort": effort},
            )
            # Prefer unified helper if available
            content = getattr(resp, "output_text", None)
            if not content:
                # Fallback to first text part
                try:
                    first = resp.output[0]
                    parts = first.content if hasattr(first, "content") else []
                    if parts and getattr(parts[0], "type", "") == "output_text":
                        content = parts[0].text
                except Exception:
                    content = None
            if not content:
                content = "{}"
            return normalize_json_content(content)
        except (RateLimitError, APIConnectionError) as e:
            if attempt >= max_retries:
                raise e
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 30)
            continue
        except APIStatusError as e:
            # Retry only 5xx
            status = getattr(e, "status_code", None)
            if status and 500 <= int(status) < 600 and attempt < max_retries:
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30)
                continue
            raise e


class JsonlWriter:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = asyncio.Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)

    async def write(self, obj: Dict[str, Any]) -> None:
        line = json.dumps(obj, ensure_ascii=False)
        async with self._lock:
            # Synchronous file I/O within async lock is acceptable here
            with self.path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
                f.flush()


async def worker(
    sem: asyncio.Semaphore,
    client: AsyncOpenAI,
    model: str,
    system_prompt: str,
    index: int,
    english: str,
    out_writer: JsonlWriter,
    fail_writer: Optional[JsonlWriter],
    progress: tqdm,
    effort: str,
) -> None:
    async with sem:
        started_at = int(time.time())
        try:
            input_text = build_input(system_prompt, english)
            result = await call_with_retries(client, model, input_text, effort)
            label = result.get("label")
            if label not in {"E0", "E1", "E2", "E3"}:
                raise ValueError(f"Invalid label returned: {label}")

            record = {
                "index": index,
                "Original_DWA_Title": english,
                "label": label,
                "explanation": result.get("explanation", ""),
                "model": model,
                "started_at": started_at,
                "completed_at": int(time.time()),
            }
            await out_writer.write(record)
        except Exception as e:
            if fail_writer is not None:
                await fail_writer.write(
                    {
                        "index": index,
                        "Original_DWA_Title": english,
                        "error": str(e),
                        "model": model,
                        "started_at": started_at,
                        "failed_at": int(time.time()),
                    }
                )
        finally:
            progress.update(1)


async def run_async(
    input_csv: Path,
    prompt_file: Path,
    output_jsonl: Path,
    failed_jsonl: Optional[Path],
    concurrency: int,
    model: str,
    limit: Optional[int] = None,
    skip_processed: bool = True,
    api_key: str = "",
    base_url: Optional[str] = None,
    sample: Optional[int] = None,
    seed: int = 42,
    effort: str = "medium",
) -> None:
    system_prompt = load_prompt(prompt_file)
    df = read_csv(input_csv)

    processed = load_processed_indices(output_jsonl) if skip_processed else set()
    total = len(df)
    # Determine indices to run considering resume and sampling
    unprocessed: List[int] = [i for i in range(total) if i not in processed]
    if sample is not None:
        k = min(sample, len(unprocessed))
        rng = random.Random(seed)
        indices_to_run: List[int] = rng.sample(unprocessed, k)
    else:
        indices_to_run = unprocessed
    if limit is not None:
        indices_to_run = indices_to_run[:limit]

    client = AsyncOpenAI(api_key=api_key, base_url=base_url)
    sem = asyncio.Semaphore(concurrency)
    out_writer = JsonlWriter(output_jsonl)
    fail_writer = JsonlWriter(failed_jsonl) if failed_jsonl is not None else None

    pbar_desc = (
        f"Processed: {len(processed)} | To run: {len(indices_to_run)} | Concurrency: {concurrency}"
    )
    with tqdm(total=len(indices_to_run), desc=pbar_desc) as progress:
        tasks: List[asyncio.Task] = []
        for idx in indices_to_run:
            row = df.iloc[idx]
            english = str(row["Original_DWA_Title"]) if not pd.isna(row["Original_DWA_Title"]) else ""
            task = asyncio.create_task(
                worker(
                    sem,
                    client,
                    model,
                    system_prompt,
                    idx,
                    english,
                    out_writer,
                    fail_writer,
                    progress,
                    effort,
                )
            )
            tasks.append(task)
        if tasks:
            await asyncio.gather(*tasks)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch classify DWA titles with OpenAI (resume + progress + parallel)",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("dwa_titles_translated.csv"),
        help="Input CSV with column Original_DWA_Title",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default="sk-eOInMs2Ay5dTX5b1engDiBGiNqbouFf7zpMEjN0xUddxbGIt",
        required=False,
        help="OpenAI API key (manual, do not use env vars)",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default="https://api3.wlai.vip/v1",
        help="OpenAI Base URL (e.g. https://api.openai.com/v1)",
    )
    parser.add_argument(
        "--prompt-file",
        type=Path,
        default=Path("oai提示词md.md"),
        help="Markdown file containing taxonomy/prompt context",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs.jsonl"),
        help="JSONL file to append results (used for resume)",
    )
    parser.add_argument(
        "--failed-output",
        type=Path,
        default=Path("failed.jsonl"),
        help="JSONL file to append failures",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=8,
        help="Number of concurrent API calls",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-5-2025-08-07",
        help="OpenAI model name",
    )
    parser.add_argument(
        "--effort",
        type=str,
        choices=["low", "medium", "high"],
        default="high",
        help="Reasoning effort for thinking-capable models (low/medium/high)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only first N unprocessed rows",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Randomly sample N unprocessed rows to run",
    )
    parser.add_argument(
        "--test10",
        action="store_true",
        help="Quick test mode: randomly sample 10 unprocessed rows",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore existing outputs and process all rows",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # Manual API key and base URL (no env vars)
    api_key = args.api_key
    if not api_key:
        raise EnvironmentError("Please provide API key via --api-key (env vars are not used).")
    base_url = args.base_url

    # Resolve sampling
    sample_count: Optional[int] = args.sample
    if args.test10:
        sample_count = 10 if sample_count is None else sample_count

    asyncio.run(
        run_async(
            input_csv=args.input,
            prompt_file=args.prompt_file,
            output_jsonl=args.output,
            failed_jsonl=args.failed_output,
            concurrency=args.concurrency,
            model=args.model,
            limit=args.limit,
            skip_processed=(not args.no_resume),
            api_key=api_key,
            base_url=base_url,
            sample=sample_count,
            seed=args.seed,
            effort=args.effort,
        )
    )


if __name__ == "__main__":
    main()


