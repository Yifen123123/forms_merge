from pathlib import Path
import json
import numpy as np
import ollama


DATASET_PATH = Path("processed/classifier_dataset.json")
CALLS_DIR = Path("calls")
OUTPUT_DIR = Path("processed/embeddings")

MODEL_NAME = "nomic-embed-text"

EMBEDDINGS_PATH = OUTPUT_DIR / "call_embeddings_nomic_safe_chunk.npz"
METADATA_PATH = OUTPUT_DIR / "embedding_metadata_nomic_safe_chunk.json"

MAX_CHARS_NO_CHUNK = 1500
CHUNK_SIZE = 1500
STRIDE = 750
FALLBACK_CHUNK_SIZE = 800


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_call_text(call_id: str) -> str:
    path = CALLS_DIR / f"{call_id}.txt"

    if not path.exists():
        raise FileNotFoundError(f"找不到通話檔案：{path}")

    text = path.read_text(encoding="utf-8").strip()

    if not text:
        raise ValueError(f"通話檔案是空的：{path}")

    return text


def get_position_weight(chunk_center_ratio: float) -> float:
    if chunk_center_ratio < 0.3:
        return 0.9
    elif chunk_center_ratio < 0.7:
        return 1.2
    else:
        return 1.0


def chunk_text_by_chars(text: str):
    if len(text) <= MAX_CHARS_NO_CHUNK:
        return [text], [1.0]

    chunks = []
    weights = []

    total_len = len(text)
    start = 0

    while start < total_len:
        end = min(start + CHUNK_SIZE, total_len)
        chunk = text[start:end]

        center = (start + end) / 2
        center_ratio = center / total_len

        chunks.append(chunk)
        weights.append(get_position_weight(center_ratio))

        if end == total_len:
            break

        start += STRIDE

    return chunks, weights


def normalize_vector(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)

    if norm == 0:
        return vector

    return vector / norm


def normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1

    return vectors / norms


def weighted_average(vectors: np.ndarray, weights: list[float]) -> np.ndarray:
    weights = np.array(weights, dtype=np.float32)
    weights = weights / weights.sum()

    return np.sum(vectors * weights[:, None], axis=0)


def embed_single_text_with_ollama(text: str) -> np.ndarray:
    response = ollama.embed(
        model=MODEL_NAME,
        input=text
    )

    vector = np.array(response["embeddings"][0], dtype=np.float32)

    if vector.ndim != 1:
        raise ValueError(f"Ollama 回傳單筆 embedding 維度異常：{vector.shape}")

    return vector


def split_fallback_chunks(text: str) -> list[str]:
    return [
        text[i:i + FALLBACK_CHUNK_SIZE]
        for i in range(0, len(text), FALLBACK_CHUNK_SIZE)
        if text[i:i + FALLBACK_CHUNK_SIZE].strip()
    ]


def embed_texts_with_ollama(texts: list[str]) -> np.ndarray:
    vectors = []

    for idx, text in enumerate(texts, start=1):
        try:
            print(f"  embedding chunk {idx}/{len(texts)}, chars={len(text)}")

            vector = embed_single_text_with_ollama(text)
            vectors.append(vector)

        except Exception as e:
            print(f"  chunk {idx} embedding 失敗，改用 fallback 切更小段")
            print(f"  chunk chars={len(text)}")
            print(f"  error={e}")

            sub_chunks = split_fallback_chunks(text)

            if not sub_chunks:
                raise ValueError("fallback 後沒有任何可 embedding 的文字")

            sub_vectors = []

            for sub_idx, sub_text in enumerate(sub_chunks, start=1):
                print(
                    f"    embedding sub-chunk {sub_idx}/{len(sub_chunks)}, "
                    f"chars={len(sub_text)}"
                )

                sub_vector = embed_single_text_with_ollama(sub_text)
                sub_vectors.append(sub_vector)

            sub_vectors = np.vstack(sub_vectors).astype("float32")
            sub_vectors = normalize_vectors(sub_vectors)

            vector = np.mean(sub_vectors, axis=0)
            vector = normalize_vector(vector)

            vectors.append(vector)

    return np.vstack(vectors).astype("float32")


def build_call_vector(text: str) -> tuple[np.ndarray, int, list[float]]:
    chunks, weights = chunk_text_by_chars(text)

    chunk_vectors = embed_texts_with_ollama(chunks)
    chunk_vectors = normalize_vectors(chunk_vectors)

    call_vector = weighted_average(chunk_vectors, weights)
    call_vector = normalize_vector(call_vector)

    return call_vector.astype("float32"), len(chunks), weights


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    dataset = load_json(DATASET_PATH)

    all_vectors = []
    all_call_ids = []
    all_labels = []
    all_doc_types = []
    all_units = []
    all_categories = []

    metadata = []

    for idx, item in enumerate(dataset, start=1):
        call_id = item["call_id"]
        label_name = item["label_name"]

        print(f"\n[{idx}/{len(dataset)}] embedding call_id={call_id}")

        text = read_call_text(call_id)

        call_vector, num_chunks, weights = build_call_vector(text)

        all_vectors.append(call_vector)
        all_call_ids.append(call_id)
        all_labels.append(label_name)
        all_doc_types.append(item.get("doc_type", ""))
        all_units.append(item.get("unit", ""))
        all_categories.append(item.get("category", ""))

        metadata.append({
            "call_id": call_id,
            "label_name": label_name,
            "doc_type": item.get("doc_type", ""),
            "unit": item.get("unit", ""),
            "category": item.get("category", ""),
            "embedding_model": MODEL_NAME,
            "chunk_method": "safe_char_chunk_with_fallback",
            "num_chunks": num_chunks,
            "chunk_weights": weights,
            "text_length_chars": len(text),
            "max_chars_no_chunk": MAX_CHARS_NO_CHUNK,
            "chunk_size": CHUNK_SIZE,
            "stride": STRIDE,
            "fallback_chunk_size": FALLBACK_CHUNK_SIZE,
        })

    vectors = np.vstack(all_vectors).astype("float32")

    np.savez(
        EMBEDDINGS_PATH,
        vectors=vectors,
        call_ids=np.array(all_call_ids),
        labels=np.array(all_labels),
        doc_types=np.array(all_doc_types),
        units=np.array(all_units),
        categories=np.array(all_categories)
    )

    with METADATA_PATH.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print("\nNomic safe chunk embedding 建立完成")
    print(f"向量檔案：{EMBEDDINGS_PATH}")
    print(f"metadata：{METADATA_PATH}")
    print(f"vectors shape: {vectors.shape}")


if __name__ == "__main__":
    main()
