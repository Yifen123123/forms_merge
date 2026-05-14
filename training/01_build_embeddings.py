from pathlib import Path
import json
import numpy as np
from sentence_transformers import SentenceTransformer


DATASET_PATH = Path("processed/classifier_dataset.json")
CALLS_DIR = Path("calls")
OUTPUT_DIR = Path("processed/embeddings")

MODEL_NAME = "intfloat/multilingual-e5-small"

CHUNK_SIZE = 400
STRIDE = 200

EMBEDDINGS_PATH = OUTPUT_DIR / "call_embeddings.npz"
METADATA_PATH = OUTPUT_DIR / "embedding_metadata.json"


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_call_text(call_id: str) -> str:
    path = CALLS_DIR / f"{call_id}.txt"

    if not path.exists():
        raise FileNotFoundError(f"找不到通話檔案：{path}")

    return path.read_text(encoding="utf-8").strip()


def get_position_weight(chunk_center_ratio: float) -> float:
    """
    方法：位置權重 position weighting

    假設客服通話：
    - 前段：開場、身分確認較多
    - 中段：主要需求最明顯
    - 後段：處理結果與確認

    權重設計：
    - prefix: 0.9
    - middle: 1.2
    - suffix: 1.0
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

    不用字元切，改用 tokenizer 切 token。
    每段最多 CHUNK_SIZE tokens，每次滑動 STRIDE tokens。
    這樣長文本不會只讀前 512 tokens。
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

        weight = get_position_weight(center_ratio)

        chunks.append(chunk_text)
        weights.append(weight)

        if end == total_tokens:
            break

        start += STRIDE

    return chunks, weights


def weighted_average(vectors: np.ndarray, weights: list[float]) -> np.ndarray:
    """
    方法：weighted average pooling

    將一通電話的多個 chunk embedding 合成一個 call embedding。
    """
    weights = np.array(weights, dtype=np.float32)
    weights = weights / weights.sum()

    return np.sum(vectors * weights[:, None], axis=0)


def normalize_vector(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)

    if norm == 0:
        return vector

    return vector / norm


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    dataset = load_json(DATASET_PATH)

    model = SentenceTransformer(MODEL_NAME)
    tokenizer = model.tokenizer

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

        print(f"[{idx}/{len(dataset)}] embedding call_id={call_id}")

        text = read_call_text(call_id)

        chunks, weights = chunk_text_by_tokens(text, tokenizer)

        # E5 文件 embedding 建議加上 passage:
        chunk_inputs = [f"passage: {chunk}" for chunk in chunks]

        chunk_vectors = model.encode(
            chunk_inputs,
            normalize_embeddings=True
        )

        call_vector = weighted_average(chunk_vectors, weights)
        call_vector = normalize_vector(call_vector)

        all_vectors.append(call_vector)
        all_call_ids.append(call_id)
        all_labels.append(label_name)
        all_doc_types.append(item.get("doc_type", ""))
        all_units.append(item.get("unit", ""))
        all_categories.append(item.get("category", ""))

        metadata.append(
            {
                "call_id": call_id,
                "label_name": label_name,
                "doc_type": item.get("doc_type", ""),
                "unit": item.get("unit", ""),
                "category": item.get("category", ""),
                "num_chunks": len(chunks),
                "chunk_weights": weights
            }
        )

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

    print("Embedding 建立完成")
    print(f"向量檔案：{EMBEDDINGS_PATH}")
    print(f"metadata：{METADATA_PATH}")
    print(f"vectors shape: {vectors.shape}")


if __name__ == "__main__":
    main()
