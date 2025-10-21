"""
Async batched invocations to OpenAI Responses API with:
- Parallel calls (configurable concurrency)
- Progress bar (tqdm with graceful fallback)
- Checkpointed resume (JSONL outputs + checkpoint JSON)

Usage examples:
  # Basic run
  uv run python oai_responses_batch.py \
    --input inputs.jsonl \
    --output outputs.jsonl \
    --checkpoint checkpoint.json \
    --model gpt-4o-mini \
    --system "You are a helpful assistant."

  # With higher concurrency and resume
  uv run python oai_responses_batch.py \
    --input inputs.jsonl \
    --output outputs.jsonl \
    --checkpoint checkpoint.json \
    --max-concurrency 12 \
    --resume

Input format (JSONL): one JSON object per line. Accepted fields:
- id (optional but recommended): unique identifier of the record
- input or prompt: the user text to send
- any other fields are carried through to output under "metadata".

Output format (JSONL): one JSON object per line, with fields:
- id: record id
- input: the sent input text
- response_text: best-effort extracted text from response
- response_raw: raw response dict (best-effort)
- error: error message if failed
- metadata: passthrough of extra fields from input

Environment:
- Requires OPENAI_API_KEY set in environment.

Dependencies:
- openai >= 1.0.0
- tqdm (optional; falls back to simple progress prints if missing)
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import os
import random
import signal
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


try:
    from openai import AsyncOpenAI  # type: ignore
except Exception as exc:  # pragma: no cover - graceful runtime import check
    print(
        "[ERROR] Failed to import openai. Install with: uv add openai (or pip install openai)",
        file=sys.stderr,
    )
    raise


# Optional tqdm progress bar
try:  # pragma: no cover
    from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover
    tqdm = None


DEFAULT_MODEL = "gpt-4o-mini"


@dataclasses.dataclass
class Record:
    id: str
    input_text: str
    metadata: Dict[str, Any]


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSONL at line: {line[:200]}...\n{e}") from e
    return items


def write_jsonl_line(path: Path, obj: Dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def load_checkpoint(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {"done_ids": [], "failed": {}}
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        # Corrupted checkpoint: start afresh but keep file backup
        backup = path.with_suffix(path.suffix + f".corrupt_{int(time.time())}")
        try:
            path.replace(backup)
        except Exception:
            pass
        return {"done_ids": [], "failed": {}}


def save_checkpoint(path: Path, done_ids: Set[str], failed: Dict[str, str]) -> None:
    temp_path = path.with_suffix(path.suffix + ".tmp")
    data = {"done_ids": sorted(done_ids), "failed": failed, "last_update": int(time.time())}
    with temp_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    temp_path.replace(path)


def coerce_records(
    raw_items: List[Dict[str, Any]], id_field: str, input_field: str
) -> List[Record]:
    records: List[Record] = []
    for idx, item in enumerate(raw_items):
        item_id: str = str(item.get(id_field, f"row_{idx}"))
        # accept both 'input' and 'prompt' for convenience
        text: Optional[str] = item.get(input_field) or item.get("prompt")
        if text is None:
            raise ValueError(f"Missing '{input_field}' or 'prompt' in record {idx}")
        # include passthrough metadata
        metadata = {k: v for k, v in item.items() if k not in {id_field, input_field, "prompt"}}
        records.append(Record(id=item_id, input_text=str(text), metadata=metadata))
    return records


def maybe_read_system(system_arg: Optional[str]) -> Optional[str]:
    if not system_arg:
        return None
    p = Path(system_arg)
    if p.exists() and p.is_file():
        return p.read_text(encoding="utf-8")
    return system_arg


def build_responses_input(system_text: Optional[str], user_text: str) -> List[Dict[str, Any]]:
    parts: List[Dict[str, Any]] = []
    if system_text:
        parts.append(
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": system_text},
                ],
            }
        )
    parts.append(
        {
            "role": "user",
            "content": [
                # Prefer the newer "input_text" type; some SDKs also accept "text"
                {"type": "input_text", "text": user_text},
            ],
        }
    )
    return parts


def extract_response_text(resp: Any) -> str:
    # Best-effort extraction compatible with multiple SDK versions
    try:
        text = getattr(resp, "output_text", None)
        if text:
            return str(text)
    except Exception:
        pass
    # Try dict-style
    try:
        if isinstance(resp, dict):
            # Attempt to traverse common structure
            output = resp.get("output")
            if output and isinstance(output, list):
                content = output[0].get("content")
                if content and isinstance(content, list):
                    maybe_text = content[0].get("text") or content[0].get("value")
                    if maybe_text:
                        return str(maybe_text)
    except Exception:
        pass
    # Try model_dump (pydantic)
    try:
        if hasattr(resp, "model_dump"):
            dumped = resp.model_dump()
            return extract_response_text(dumped)
    except Exception:
        pass
    # Fallback to string
    return str(resp)


def response_to_dict(resp: Any) -> Dict[str, Any]:
    # Try model_dump_json or model_dump
    try:
        if hasattr(resp, "model_dump_json"):
            return json.loads(resp.model_dump_json())  # type: ignore
    except Exception:
        pass
    try:
        if hasattr(resp, "model_dump"):
            return resp.model_dump()  # type: ignore
    except Exception:
        pass
    try:
        if hasattr(resp, "to_dict"):
            return resp.to_dict()  # type: ignore
    except Exception:
        pass
    try:
        if hasattr(resp, "json"):
            return json.loads(resp.json())  # type: ignore
    except Exception:
        pass
    # If all else fails, store repr
    return {"repr": repr(resp)}


class GracefulKiller:
    def __init__(self) -> None:
        self._stop = False
        try:
            signal.signal(signal.SIGINT, self._handler)  # Ctrl+C
            signal.signal(signal.SIGTERM, self._handler)
        except Exception:
            # Not all platforms allow signal setup
            pass

    def _handler(self, *_: Any) -> None:
        self._stop = True

    @property
    def stopped(self) -> bool:
        return self._stop


class ProgressManager:
    def __init__(self, total: int) -> None:
        self.total = total
        self.count = 0
        self._lock = asyncio.Lock()
        self._pbar = None
        if tqdm is not None:
            self._pbar = tqdm(total=total, desc="Processing", unit="req")

    async def update(self, n: int = 1) -> None:
        async with self._lock:
            self.count += n
            if self._pbar is not None:
                self._pbar.update(n)
            else:
                # Simple fallback logging every 10 updates or at the end
                if self.count % 10 == 0 or self.count == self.total:
                    print(f"Progress: {self.count}/{self.total}")

    def close(self) -> None:
        if self._pbar is not None:
            self._pbar.close()


async def bounded_call(
    sem: asyncio.Semaphore,
    fn,
    *args,
    **kwargs,
):
    async with sem:
        return await fn(*args, **kwargs)


async def send_response_request(
    client: AsyncOpenAI,
    model: str,
    system_text: Optional[str],
    user_text: str,
    temperature: float,
    max_retries: int,
) -> Any:
    last_err: Optional[Exception] = None
    for attempt in range(max_retries + 1):
        try:
            # Note: timeout is supported on the client transport; we pass per-request timeout if available.
            resp = await client.responses.create(
                model=model,
                input=build_responses_input(system_text, user_text),
                temperature=temperature,
            )
            return resp
        except Exception as e:  # Broad catch to keep the batch running
            last_err = e
            # Exponential backoff with jitter
            sleep_s = min(60.0, (2 ** attempt) + random.uniform(0, 1))
            await asyncio.sleep(sleep_s)
    assert last_err is not None
    raise last_err


async def process_record(
    record: Record,
    client: AsyncOpenAI,
    model: str,
    system_text: Optional[str],
    temperature: float,
    max_retries: int,
) -> Tuple[str, Dict[str, Any]]:
    resp = await send_response_request(
        client=client,
        model=model,
        system_text=system_text,
        user_text=record.input_text,
        temperature=temperature,
        max_retries=max_retries,
    )
    resp_text = extract_response_text(resp)
    resp_raw = response_to_dict(resp)
    out_obj = {
        "id": record.id,
        "input": record.input_text,
        "response_text": resp_text,
        "response_raw": resp_raw,
        "metadata": record.metadata,
    }
    return record.id, out_obj


async def run_batch(
    records: List[Record],
    output_path: Path,
    checkpoint_path: Path,
    model: str,
    system_text: Optional[str],
    max_concurrency: int,
    temperature: float,
    timeout_s: float,
    max_retries: int,
    resume: bool,
) -> Tuple[int, int]:
    # Load checkpoint and determine remaining
    checkpoint = load_checkpoint(checkpoint_path) if resume else {"done_ids": [], "failed": {}}
    done_ids: Set[str] = set(checkpoint.get("done_ids", []))
    failed: Dict[str, str] = dict(checkpoint.get("failed", {}))

    remaining = [r for r in records if r.id not in done_ids]
    total = len(remaining)
    if total == 0:
        print("Nothing to do. All records already processed.")
        return 0, 0

    progress = ProgressManager(total=total)
    killer = GracefulKiller()
    sem = asyncio.Semaphore(max(1, max_concurrency))
    output_lock = asyncio.Lock()
    checkpoint_lock = asyncio.Lock()

    client = AsyncOpenAI(timeout=timeout_s)

    async def handle_one(rec: Record) -> None:
        if killer.stopped:
            return
        try:
            _, obj = await bounded_call(
                sem,
                process_record,
                rec,
                client,
                model,
                system_text,
                temperature,
                max_retries,
            )
            # write output and checkpoint
            async with output_lock:
                write_jsonl_line(output_path, obj)
            async with checkpoint_lock:
                done_ids.add(rec.id)
                save_checkpoint(checkpoint_path, done_ids, failed)
        except Exception as e:
            async with checkpoint_lock:
                failed[rec.id] = str(e)
                save_checkpoint(checkpoint_path, done_ids, failed)
        finally:
            await progress.update(1)

    try:
        await asyncio.gather(*(handle_one(r) for r in remaining))
    finally:
        progress.close()

    return len(done_ids), len(failed)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Batch call OpenAI Responses API with async parallelism and resume.")
    p.add_argument("--input", required=True, help="Path to input JSONL file.")
    p.add_argument("--output", required=True, help="Path to output JSONL file.")
    p.add_argument("--checkpoint", required=True, help="Path to checkpoint JSON file.")
    p.add_argument("--model", default=DEFAULT_MODEL, help=f"Model name (default: {DEFAULT_MODEL}).")
    p.add_argument("--system", default=None, help="System instruction string or path to a file.")
    p.add_argument("--id-field", default="id", help="Field name for record id in input JSONL (default: id).")
    p.add_argument("--input-field", default="input", help="Field name for user text in input JSONL (default: input).")
    p.add_argument("--max-concurrency", type=int, default=8, help="Max parallel requests (default: 8).")
    p.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature (default: 0.2).")
    p.add_argument("--timeout", type=float, default=120.0, help="Per-request timeout seconds (default: 120).")
    p.add_argument("--retries", type=int, default=5, help="Max retries per request (default: 5).")
    p.add_argument("--resume", action="store_true", help="Resume from checkpoint; skip already-done ids.")
    p.add_argument("--overwrite", action="store_true", help="Overwrite output and checkpoint before run.")
    return p.parse_args(argv)


def prepare_paths(output_path: Path, checkpoint_path: Path, overwrite: bool) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    if overwrite:
        if output_path.exists():
            output_path.unlink()
        if checkpoint_path.exists():
            checkpoint_path.unlink()


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    in_path = Path(args.input)
    out_path = Path(args.output)
    ckpt_path = Path(args.checkpoint)

    prepare_paths(out_path, ckpt_path, overwrite=args.overwrite)

    raw_items = read_jsonl(in_path)
    records = coerce_records(raw_items, id_field=args.id_field, input_field=args.input_field)
    system_text = maybe_read_system(args.system)

    if not os.getenv("OPENAI_API_KEY"):
        print("[ERROR] OPENAI_API_KEY not set in environment.", file=sys.stderr)
        return 2

    print(
        f"Starting batch: {len(records)} records | model={args.model} | concurrency={args.max_concurrency} | resume={args.resume}")

    try:
        done_count, fail_count = asyncio.run(
            run_batch(
                records=records,
                output_path=out_path,
                checkpoint_path=ckpt_path,
                model=args.model,
                system_text=system_text,
                max_concurrency=args.max_concurrency,
                temperature=args.temperature,
                timeout_s=args.timeout,
                max_retries=args.retries,
                resume=bool(args.resume),
            )
        )
    except KeyboardInterrupt:
        print("Interrupted. Progress saved to checkpoint.")
        return 130

    print(f"Done. Completed={done_count}, Failed={fail_count}. Output: {out_path}")
    if fail_count > 0:
        print(f"Some records failed. See checkpoint: {ckpt_path}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


