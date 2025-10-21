from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple
import json
from json import JSONDecodeError

import pandas as pd

try:
    from json_repair import repair_json  # type: ignore
except Exception:  # pragma: no cover
    repair_json = None  # type: ignore


def extract_json_substring(text: str) -> str | None:
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]
    return None


def parse_json_line(line: str) -> Tuple[Dict[str, Any] | None, str | None]:
    raw = line.strip()
    if not raw:
        return None, None

    if raw.startswith("data:"):
        raw = raw[5:].strip()

    try:
        return json.loads(raw), None
    except JSONDecodeError as e1:
        candidate = extract_json_substring(raw)
        if candidate:
            try:
                return json.loads(candidate), None
            except JSONDecodeError as e2:
                return None, f"JSONDecodeError after substring extract: {e2} | line sample: {raw[:200]}"
        return None, f"JSONDecodeError: {e1} | line sample: {raw[:200]}"


def load_jsonl(path: Path) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    records: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []

    with path.open("r", encoding="utf-8-sig", errors="replace") as f:
        for idx, line in enumerate(f, start=1):
            obj, err = parse_json_line(line)
            if obj is not None:
                records.append(obj)
            elif err:
                errors.append({"line_no": idx, "error": err})
    return records, errors


def parse_content_to_responsibilities(content: Any) -> Tuple[List[str] | None, str | None]:
    if content is None:
        return None, "no_content"

    if isinstance(content, dict):
        if "responsibilities" in content and isinstance(content["responsibilities"], list):
            items = [str(x).strip() for x in content["responsibilities"]]
            items = [x for x in items if x]
            return items, None
        if "content" in content:
            return parse_content_to_responsibilities(content["content"])
        return None, "dict_without_responsibilities"

    if isinstance(content, str):
        s = content.strip()
        try:
            loaded = json.loads(s)
        except Exception:
            candidate = extract_json_substring(s)
            loaded = None
            if candidate:
                try:
                    loaded = json.loads(candidate)
                except Exception:
                    loaded = None
            if loaded is None and repair_json is not None:
                try:
                    to_repair = candidate if candidate else s
                    repaired = repair_json(to_repair)
                    loaded = json.loads(repaired)
                except Exception as e_repair:
                    return None, f"json_repair_failed:{e_repair}"
            elif loaded is None:
                return None, "json_parse_failed"
        return parse_content_to_responsibilities(loaded)

    return None, f"unsupported_content_type:{type(content).__name__}"


def parse_responsibilities_from_body(body: Any) -> Tuple[List[str] | None, str | None]:
    if not isinstance(body, dict):
        return None, "body_not_dict"

    content = None
    choices = body.get("choices")
    if isinstance(choices, list) and len(choices) > 0 and isinstance(choices[0], dict):
        first = choices[0]
        msg = first.get("message")
        if isinstance(msg, dict):
            content = msg.get("content")
        if content is None:
            content = first.get("content")
        if content is None and isinstance(first.get("delta"), dict):
            content = first["delta"].get("content")

    if content is None:
        message_field = body.get("message") if isinstance(body.get("message"), dict) else {}
        content = (
            body.get("output_text")
            or body.get("content")
            or (message_field.get("content") if isinstance(message_field, dict) else None)
        )

    return parse_content_to_responsibilities(content)


def process_one_jsonl(path: Path) -> pd.DataFrame:
    records, _ = load_jsonl(path)
    if not records:
        return pd.DataFrame({"file": [path.name], "parse_exception": [None]})

    df_one = pd.json_normalize(records, max_level=1)
    if "response.body" in df_one.columns:
        parsed_series = df_one["response.body"].apply(parse_responsibilities_from_body)
        df_one["responsibilities"] = parsed_series.map(lambda t: t[0])
        df_one["responsibilities_error"] = parsed_series.map(lambda t: t[1])

    df_one["file"] = path.name
    keep_cols = [
        c for c in [
            "custom_id",
            "id",
            "response.status_code",
            "responsibilities",
            "responsibilities_error",
            "file",
        ]
        if c in df_one.columns
    ]
    return df_one[keep_cols]


def main() -> None:
    project_root = Path(__file__).resolve().parent
    jsonl_dir = project_root / "中间文件" / "batch_return_files"
    assert jsonl_dir.exists(), f"目录不存在: {jsonl_dir}"

    files = sorted(jsonl_dir.glob("*.jsonl"))
    print(f"Found {len(files)} JSONL files under {jsonl_dir}")

    frames: List[pd.DataFrame] = []
    for p in files:
        try:
            frames.append(process_one_jsonl(p))
        except Exception as e:
            frames.append(pd.DataFrame({"file": [p.name], "parse_exception": [str(e)]}))

    merged = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
    print("Merged shape:", merged.shape)
    if merged.empty:
        print("No data to export.")
        return

    def _is_non_empty_list(x: Any) -> bool:
        return isinstance(x, list) and len(x) > 0

    merged["is_list"] = merged.get("responsibilities", pd.Series([None] * len(merged))).apply(_is_non_empty_list)
    merged["len_list"] = merged.get("responsibilities", pd.Series([None] * len(merged))).apply(
        lambda x: len(x) if isinstance(x, list) else -1
    )

    key_col = "custom_id" if "custom_id" in merged.columns else ("id" if "id" in merged.columns else None)
    if key_col is None:
        key_col = "_row_index"
        merged[key_col] = merged.index.astype(str)

    status = merged.get("response.status_code", pd.Series([-1] * len(merged)))
    merged["_is_ok"] = status.eq(200)

    merged_sorted = merged.sort_values(by=["_is_ok", "is_list", "len_list"], ascending=[False, False, False])
    best_rows = merged_sorted.drop_duplicates(subset=[key_col], keep="first").copy()

    parsed_only = best_rows[best_rows["is_list"]].copy()

    export_cols = [
        c
        for c in [key_col, "custom_id", "id", "responsibilities", "responsibilities_error", "response.status_code", "file"]
        if c in best_rows.columns
    ]

    out_dir = project_root / "中间文件"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_parsed_csv = out_dir / "batch_return_merged.csv"
    out_parsed_parquet = out_dir / "batch_return_merged.parquet"
    out_best_all_parquet = out_dir / "batch_return_merged_all.parquet"

    print(f"Saving parsed-only CSV -> {out_parsed_csv}")
    parsed_only[export_cols].to_csv(out_parsed_csv, index=False, encoding="utf-8-sig")

    print(f"Saving parsed-only Parquet -> {out_parsed_parquet}")
    parsed_only[export_cols].to_parquet(out_parsed_parquet, index=False)

    print(f"Saving best-all Parquet -> {out_best_all_parquet}")
    best_rows[export_cols].to_parquet(out_best_all_parquet, index=False)

    print(
        f"Done. parsed-only rows: {len(parsed_only)} / best-all rows: {len(best_rows)} | key_col: {key_col}"
    )


if __name__ == "__main__":
    main()





















