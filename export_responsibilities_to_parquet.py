import argparse
import json
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import pandas as pd


def extract_record(line: str) -> Optional[Tuple[str, List[str]]]:
    try:
        obj = json.loads(line)
    except Exception:
        return None

    custom_id = obj.get("custom_id")
    if not custom_id:
        return None

    content = None
    try:
        # Zhipu/OpenAI batch style
        content = (
            obj["response"]["body"]["choices"][0]["message"]["content"]
        )
    except Exception:
        pass

    responsibilities: List[str] = []
    if isinstance(content, str):
        try:
            payload = json.loads(content)
            cand = payload.get("responsibilities")
            if isinstance(cand, list):
                responsibilities = [str(x) for x in cand if isinstance(x, (str, int, float))]
        except Exception:
            responsibilities = []

    return custom_id, responsibilities


def iter_jsonl(jsonl_paths: Iterable[Path]) -> Iterable[Tuple[str, List[str]]]:
    for p in jsonl_paths:
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = extract_record(line)
                if rec is not None:
                    yield rec


def main():
    parser = argparse.ArgumentParser(description="Export custom_id and responsibilities to Parquet")
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="Input JSONL files or directories containing JSONL",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output Parquet file path",
    )
    args = parser.parse_args()

    input_paths: List[Path] = []
    for ip in args.inputs:
        p = Path(ip)
        if p.is_dir():
            input_paths.extend(sorted(p.glob("*.jsonl")))
        elif p.is_file():
            input_paths.append(p)

    records = list(iter_jsonl(input_paths))
    df = pd.DataFrame(records, columns=["custom_id", "responsibilities"])

    # Ensure stable ordering by custom_id
    if not df.empty:
        df = df.sort_values("custom_id").reset_index(drop=True)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"Wrote {len(df)} rows to {out_path}")


if __name__ == "__main__":
    main()


