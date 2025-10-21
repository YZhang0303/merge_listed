import argparse
from pathlib import Path
from typing import List, Sequence

import pandas as pd


def embed_texts(texts: Sequence[str], model: str) -> List[List[float]]:
    # Placeholder: replace with your remote host's embedding API call.
    # For example, using sentence-transformers or OpenAI-compatible API.
    # Here we create deterministic faux embeddings by hashing.
    def _hash_to_vec(s: str, dim: int = 384) -> List[float]:
        h = abs(hash(s))
        return [((h >> (i % 32)) & 1) * 1.0 for i in range(dim)]

    return [_hash_to_vec(t or "") for t in texts]


def join_responsibilities(resps: List[str]) -> str:
    if not isinstance(resps, list):
        return ""
    return " \n".join(str(x) for x in resps if isinstance(x, (str, int, float)))


def main():
    parser = argparse.ArgumentParser(description="Embed responsibilities from Parquet and write back to Parquet")
    parser.add_argument("--input", required=True, help="Input Parquet with columns: custom_id, responsibilities")
    parser.add_argument("--output", required=True, help="Output Parquet path with embeddings")
    parser.add_argument("--model", default="bge-m3", help="Embedding model to use on remote host")
    args = parser.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(in_path)
    if "custom_id" not in df.columns or "responsibilities" not in df.columns:
        raise ValueError("Input parquet must contain columns: custom_id, responsibilities")

    # Join responsibilities into a single text per row for embedding
    texts = [join_responsibilities(x) for x in df["responsibilities"].tolist()]
    embeddings = embed_texts(texts, model=args.model)

    out_df = pd.DataFrame({
        "custom_id": df["custom_id"].tolist(),
        "responsibilities": df["responsibilities"].tolist(),
        "embedding": embeddings,
    })

    out_df.to_parquet(out_path, index=False)
    print(f"Wrote embeddings for {len(out_df)} rows to {out_path}")


if __name__ == "__main__":
    main()


