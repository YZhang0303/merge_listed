import argparse
import json
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import pandas as pd


def _parse_responsibilities_from_content(content: object) -> List[str]:
    if isinstance(content, list):
        return [str(x) for x in content]
    if isinstance(content, dict):
        cand = content.get("responsibilities")
        if isinstance(cand, list):
            return [str(x) for x in cand]
        return []
    if isinstance(content, str):
        try:
            obj = json.loads(content)
        except Exception:
            return []
        return _parse_responsibilities_from_content(obj)
    return []


def _extract_from_jsonl_line(line: str) -> Optional[Tuple[str, List[str]]]:
    try:
        obj = json.loads(line)
    except Exception:
        return None

    custom_id = obj.get("custom_id")
    if not custom_id:
        return None

    content = None
    # Try OpenAI/Zhipu batch response shapes
    try:
        content = obj["response"]["body"]["choices"][0]["message"].get("content")
    except Exception:
        pass
    if content is None:
        try:
            content = obj["response"]["body"].get("content")
        except Exception:
            pass

    responsibilities = _parse_responsibilities_from_content(content)
    return custom_id, responsibilities


def iter_jsonl(paths: Sequence[Path]) -> Iterable[Tuple[str, List[str]]]:
    for p in paths:
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                rec = _extract_from_jsonl_line(line)
                if rec is not None:
                    yield rec


def load_records(inputs: Sequence[str]) -> List[Tuple[str, List[str]]]:
    input_paths: List[Path] = []
    for ip in inputs:
        p = Path(ip)
        if p.is_dir():
            input_paths.extend(sorted(p.glob("*.jsonl")))
        elif p.suffix.lower() == ".jsonl":
            input_paths.append(p)
        elif p.suffix.lower() in {".parquet"}:
            # Parquet with columns: custom_id, responsibilities
            df = pd.read_parquet(p)
            if "custom_id" in df.columns and "responsibilities" in df.columns:
                return list(zip(df["custom_id"].astype(str).tolist(), df["responsibilities"].tolist()))
    # Fallback: read JSONL files
    return list(iter_jsonl(input_paths))


def to_long_dataframe(records: Sequence[Tuple[str, List[str]]]) -> pd.DataFrame:
    rows: List[Tuple[str, int, str]] = []
    for custom_id, items in records:
        if not isinstance(items, list):
            continue
        for idx, text in enumerate(items):
            rows.append((str(custom_id), idx, str(text)))
    df_long = pd.DataFrame(rows, columns=["custom_id", "item_idx", "responsibility"])
    # Stable ordering by custom_id then item_idx
    if not df_long.empty:
        df_long = df_long.sort_values(["custom_id", "item_idx"]).reset_index(drop=True)
        df_long.insert(0, "line_no", range(len(df_long)))
    else:
        df_long.insert(0, "line_no", [])
    return df_long


def main():
    parser = argparse.ArgumentParser(description="Flatten responsibilities to a long list with stable line_no")
    parser.add_argument("--inputs", nargs="+", required=True, help="Input Parquet or JSONL files/dirs")
    parser.add_argument("--out_csv", required=False, help="Output CSV path")
    parser.add_argument("--out_parquet", required=True, help="Output Parquet path")
    args = parser.parse_args()

    records = load_records(args.inputs)
    df_long = to_long_dataframe(records)

    out_parquet = Path(args.out_parquet)
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    df_long.to_parquet(out_parquet, index=False)

    if args.out_csv:
        out_csv = Path(args.out_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        df_long.to_csv(out_csv, index=False, encoding="utf-8-sig")

    print(f"Wrote {len(df_long)} lines to {out_parquet}")


if __name__ == "__main__":
    main()





