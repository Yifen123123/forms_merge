from pathlib import Path
import argparse
import json
import numpy as np
import joblib
import faiss
import ollama


MODEL_NAME = "nomic-embed-text"

SVM_MODEL_PATH = Path("models/svm_classifier_nomic.joblib")
LABEL_ENCODER_PATH = Path("models/label_encoder_nomic.joblib")

FAISS_INDEX_PATH = Path("processed/faiss/call_vectors_nomic.index")
FAISS_METADATA_PATH = Path("processed/faiss/faiss_metadata_nomic.json")

TOP_K = 5

MAX_CHARS_NO_CHUNK = 1500
CHUNK_SIZE = 1500
STRIDE = 750
FALLBACK_CHUNK_SIZE = 800


def get_position_weight(chunk_center_ratio: float) -> float:
    if chunk_center_ratio < 0.3:
        return 0.9
    elif chunk_center_ratio < 0.7:
        return 1.2
    else:
        return 1.0


def chunk_text_by_chars(text: str) -> tuple[list[str], list[float]]:
    if len(text) <= MAX_CHARS_NO_CHUNK:
        return [text], [1.0]

    chunks = []
    weights = []

    total_len = len(text)
    start = 0

    while start < total_len:
        end = min(start + CHUNK_SIZE, total_len)
        chunk = text[start:end].strip()

        if chunk:
            center = (start + end) / 2
            center_ratio = center / total_len

            chunks.append(chunk)
            weights.append(get_position_weight(center_ratio))

        if end == total_len:
            break

        start += STRIDE

    return chunks, weights


def split_fallback_chunks(text: str) -> list[str]:
    return [
        text[i:i + FALLBACK_CHUNK_SIZE].strip()
        for i in range(0, len(text), FALLBACK_CHUNK_SIZE)
        if text[i:i + FALLBACK_CHUNK_SIZE].strip()
    ]


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


def build_call_vector(text: str) -> tuple[np.ndarray, int]:
    chunks, weights = chunk_text_by_chars(text)

    if not chunks:
        raise ValueError("輸入文字切 chunk 後沒有任何內容")

    chunk_vectors = embed_texts_with_ollama(chunks)
    chunk_vectors = normalize_vectors(chunk_vectors)

    call_vector = weighted_average(chunk_vectors, weights)
    call_vector = normalize_vector(call_vector)

    return call_vector.astype("float32"), len(chunks)


def predict_with_svm(vector: np.ndarray, clf, label_encoder):
    X = vector.reshape(1, -1)

    pred_encoded = clf.predict(X)[0]
    pred_label = label_encoder.inverse_transform([pred_encoded])[0]

    confidence = None

    if hasattr(clf, "predict_proba"):
        proba = clf.predict_proba(X)[0]
        confidence = float(np.max(proba))

    return pred_label, confidence


def search_similar_cases(vector: np.ndarray, top_k: int):
    index = faiss.read_index(str(FAISS_INDEX_PATH))

    with FAISS_METADATA_PATH.open("r", encoding="utf-8") as f:
        faiss_metadata = json.load(f)

    items = faiss_metadata["items"]

    query_vector = normalize_vector(vector).reshape(1, -1).astype("float32")

    scores, indices = index.search(query_vector, top_k)

    results = []

    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue

        item = items[idx]

        results.append({
            "call_id": item["call_id"],
            "label_name": item["label_name"],
            "doc_type": item.get("doc_type", ""),
            "unit": item.get("unit", ""),
            "category": item.get("category", ""),
            "similarity": float(score),
            "num_chunks": item.get("num_chunks"),
            "embedding_model": item.get("embedding_model", MODEL_NAME),
        })

    return results


def parse_args():
    parser = argparse.ArgumentParser(
        description="Predict meeting form category for a new call text using Nomic Embed."
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
        help="FAISS 相似案例數量，預設 5"
    )

    parser.add_argument(
        "--output",
        default=None,
        help="可選：輸出 JSON 結果路徑，例如 processed/predictions/result_nomic.json"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    input_path = Path(args.input)

    if not input_path.exists():
        raise FileNotFoundError(f"找不到新通話檔案：{input_path}")

    text = input_path.read_text(encoding="utf-8").strip()

    if not text:
        raise ValueError("新通話檔案是空的")

    clf = joblib.load(SVM_MODEL_PATH)
    label_encoder = joblib.load(LABEL_ENCODER_PATH)

    print(f"讀取新通話：{input_path}")
    print(f"文字長度：{len(text)} chars")

    vector, num_chunks = build_call_vector(text)

    pred_label, confidence = predict_with_svm(
        vector=vector,
        clf=clf,
        label_encoder=label_encoder
    )

    similar_cases = search_similar_cases(
        vector=vector,
        top_k=args.top_k
    )

    result = {
        "input_file": str(input_path),
        "embedding_model": MODEL_NAME,
        "chunk_method": "safe_char_chunk_with_fallback",
        "num_chunks": num_chunks,
        "text_length_chars": len(text),
        "svm_prediction": {
            "label_name": pred_label,
            "confidence": confidence
        },
        "top_similar_cases": similar_cases
    }

    print(json.dumps(result, ensure_ascii=False, indent=2))

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with output_path.open("w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        print(f"\n預測結果已儲存至：{output_path}")


if __name__ == "__main__":
    main()
