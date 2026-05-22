from pathlib import Path
import argparse
import json
import time
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


MODEL_NAME = "intfloat/multilingual-e5-small"

INDEX_PATH = Path("processed/category_faiss/category_rules.index")
METADATA_PATH = Path("processed/category_faiss/category_rules_metadata.json")

TOP_K = 5
MAX_CALL_CHARS = 6000


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_text(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"找不到通話檔案：{path}")

    text = path.read_text(encoding="utf-8").strip()

    if not text:
        raise ValueError(f"通話檔案是空的：{path}")

    return text[:MAX_CALL_CHARS]


def normalize_vector(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)

    if norm == 0:
        return vector

    return vector / norm


def retrieve_categories(call_text: str, top_k: int):
    timing = {}

    start = time.perf_counter()
    model = SentenceTransformer(MODEL_NAME)
    timing["load_model_seconds"] = time.perf_counter() - start

    start = time.perf_counter()
    index = faiss.read_index(str(INDEX_PATH))
    metadata = load_json(METADATA_PATH)
    timing["load_index_and_metadata_seconds"] = time.perf_counter() - start

    start = time.perf_counter()
    query_text = f"query: {call_text}"

    query_vector = model.encode(
        [query_text],
        normalize_embeddings=True
    ).astype("float32")

    query_vector[0] = normalize_vector(query_vector[0])
    timing["embedding_seconds"] = time.perf_counter() - start

    start = time.perf_counter()
    scores, indices = index.search(query_vector, top_k)
    timing["faiss_search_seconds"] = time.perf_counter() - start

    start = time.perf_counter()
    results = []

    for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), start=1):
        if idx == -1:
            continue

        item = metadata[idx]

        results.append({
            "rank": rank,
            "score": float(score),
            "category_index": item["category_index"],
            "label_name": item["label_name"],
            "doc_type": item["doc_type"],
            "unit": item["unit"],
            "category": item["category"],
            "definition": item.get("definition", ""),
            "num_examples": item.get("num_examples", 0),
            "data_sufficiency": item.get("data_sufficiency", "")
        })

    timing["format_results_seconds"] = time.perf_counter() - start
    timing["total_retrieval_seconds"] = sum(timing.values())

    return results, timing


def parse_args():
    parser = argparse.ArgumentParser(
        description="Test category rule retrieval using FAISS and measure execution time."
    )

    parser.add_argument(
        "--input",
        required=True,
        help="新通話 txt 檔案路徑，例如 new_calls/new_call.txt"
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=TOP_K,
        help="取回前幾個候選分類，預設 5"
    )

    parser.add_argument(
        "--output",
        default=None,
        help="可選：輸出 JSON 路徑，例如 processed/retrieval/test_result.json"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    total_start = time.perf_counter()

    start = time.perf_counter()
    call_text = read_text(Path(args.input))
    read_input_seconds = time.perf_counter() - start

    results, timing = retrieve_categories(
        call_text=call_text,
        top_k=args.top_k
    )

    total_elapsed = time.perf_counter() - total_start

    timing["read_input_seconds"] = read_input_seconds
    timing["total_script_seconds"] = total_elapsed

    output = {
        "input_file": args.input,
        "top_k": args.top_k,
        "model_name": MODEL_NAME,
        "call_text_chars": len(call_text),
        "timing": {
            key: round(value, 6)
            for key, value in timing.items()
        },
        "retrieved_categories": results
    }

    print(json.dumps(output, ensure_ascii=False, indent=2))

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with output_path.open("w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)

        print(f"\n結果已儲存至：{output_path}")


if __name__ == "__main__":
    main()
