from pathlib import Path
import argparse
import json
import numpy as np
import joblib
import faiss
from sentence_transformers import SentenceTransformer


MODEL_NAME = "intfloat/multilingual-e5-small"

SVM_MODEL_PATH = Path("models/svm_classifier.joblib")
LABEL_ENCODER_PATH = Path("models/label_encoder.joblib")

FAISS_INDEX_PATH = Path("processed/faiss/call_vectors.index")
FAISS_METADATA_PATH = Path("processed/faiss/faiss_metadata.json")

CHUNK_SIZE = 400
STRIDE = 200
TOP_K = 5


def get_position_weight(chunk_center_ratio: float) -> float:
    """
    方法：位置權重 position weighting
    與 01_build_embeddings.py 保持一致。
    """
    if chunk_center_ratio < 0.3:
        return 0.9
    elif chunk_center_ratio < 0.7:
        return 1.2
    else:
        return 1.0


def chunk_text_by_tokens(text: str, tokenizer):
    """
    方法：tokenizer-based sliding window
    與訓練階段完全一致：
    - chunk_size = 400 tokens
    - stride = 200 tokens
    """
    token_ids = tokenizer.encode(
        text,
        add_special_tokens=False,
        truncation=False
    )

    if len(token_ids) <= CHUNK_SIZE:
        return [text], [1.0]

    chunks = []
    weights = []

    total_tokens = len(token_ids)
    start = 0

    while start < total_tokens:
        end = min(start + CHUNK_SIZE, total_tokens)
        chunk_ids = token_ids[start:end]

        chunk_text = tokenizer.decode(
            chunk_ids,
            skip_special_tokens=True
        )

        center = (start + end) / 2
        center_ratio = center / total_tokens

        chunks.append(chunk_text)
        weights.append(get_position_weight(center_ratio))

        if end == total_tokens:
            break

        start += STRIDE

    return chunks, weights


def weighted_average(vectors: np.ndarray, weights: list[float]) -> np.ndarray:
    """
    方法：weighted average pooling
    將多個 chunk embedding 合併成單一通話向量。
    """
    weights = np.array(weights, dtype=np.float32)
    weights = weights / weights.sum()

    return np.sum(vectors * weights[:, None], axis=0)


def normalize_vector(vector: np.ndarray) -> np.ndarray:
    """
    方法：L2 normalize
    讓 FAISS IndexFlatIP 可以等價於 cosine similarity。
    """
    norm = np.linalg.norm(vector)

    if norm == 0:
        return vector

    return vector / norm


def build_call_vector(text: str, model: SentenceTransformer) -> tuple[np.ndarray, int]:
    """
    方法：文字 → chunks → E5 embeddings → weighted pooling → call vector
    """
    tokenizer = model.tokenizer

    chunks, weights = chunk_text_by_tokens(text, tokenizer)

    # E5 文件向量建議使用 passage:
    chunk_inputs = [f"passage: {chunk}" for chunk in chunks]

    chunk_vectors = model.encode(
        chunk_inputs,
        normalize_embeddings=True
    )

    call_vector = weighted_average(chunk_vectors, weights)
    call_vector = normalize_vector(call_vector)

    return call_vector.astype("float32"), len(chunks)


def predict_with_svm(vector: np.ndarray, clf, label_encoder):
    """
    方法：Linear SVM classification
    輸入單一通話向量，輸出預測 label 與機率。
    """
    X = vector.reshape(1, -1)

    pred_encoded = clf.predict(X)[0]
    pred_label = label_encoder.inverse_transform([pred_encoded])[0]

    confidence = None

    if hasattr(clf, "predict_proba"):
        proba = clf.predict_proba(X)[0]
        confidence = float(np.max(proba))

    return pred_label, confidence


def search_similar_cases(vector: np.ndarray, top_k: int):
    """
    方法：FAISS similarity search
    找出歷史資料中最相似的通話案例。
    """
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

        results.append(
            {
                "call_id": item["call_id"],
                "label_name": item["label_name"],
                "doc_type": item.get("doc_type", ""),
                "unit": item.get("unit", ""),
                "category": item.get("category", ""),
                "similarity": float(score),
                "num_chunks": item.get("num_chunks")
            }
        )

    return results


def parse_args():
    parser = argparse.ArgumentParser(
        description="Predict meeting form category for a new call text."
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
        help="可選：輸出 JSON 結果路徑，例如 processed/predictions/result.json"
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

    model = SentenceTransformer(MODEL_NAME)

    vector, num_chunks = build_call_vector(text, model)

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
        "num_chunks": num_chunks,
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
