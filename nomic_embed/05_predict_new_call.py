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


def normalize_vector(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)

    if norm == 0:
        return vector

    return vector / norm


def build_call_vector(text: str) -> tuple[np.ndarray, int]:
    response = ollama.embed(
        model=MODEL_NAME,
        input=text
    )

    vector = np.array(
        response["embeddings"][0],
        dtype=np.float32
    )

    vector = normalize_vector(vector)

    return vector.astype("float32"), 1


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
            "num_chunks": item.get("num_chunks", 1),
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
        "chunk_method": "no_chunk_full_text",
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
