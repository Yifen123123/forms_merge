from pathlib import Path
import json
import numpy as np
import faiss


EMBEDDINGS_PATH = Path("processed/embeddings/call_embeddings.npz")
METADATA_PATH = Path("processed/embeddings/embedding_metadata.json")

OUTPUT_DIR = Path("processed/faiss")
FAISS_INDEX_PATH = OUTPUT_DIR / "call_vectors.index"
FAISS_METADATA_PATH = OUTPUT_DIR / "faiss_metadata.json"


def normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    """
    方法：L2 normalize

    FAISS IndexFlatIP 使用 inner product。
    如果向量已經 normalize，inner product 就等價於 cosine similarity。
    """
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1

    return vectors / norms


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    data = np.load(EMBEDDINGS_PATH, allow_pickle=True)

    vectors = data["vectors"].astype("float32")
    vectors = normalize_vectors(vectors).astype("float32")

    with METADATA_PATH.open("r", encoding="utf-8") as f:
        metadata = json.load(f)

    dim = vectors.shape[1]

    # 方法：FAISS IndexFlatIP
    # 用於 cosine similarity search。
    # 適合第一版 baseline，資料量不大時最直觀。
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)

    faiss.write_index(index, str(FAISS_INDEX_PATH))

    faiss_metadata = {
        "index_type": "IndexFlatIP",
        "similarity": "cosine_similarity_after_l2_normalization",
        "num_vectors": int(vectors.shape[0]),
        "dimension": int(dim),
        "items": metadata
    }

    with FAISS_METADATA_PATH.open("w", encoding="utf-8") as f:
        json.dump(faiss_metadata, f, ensure_ascii=False, indent=2)

    print("FAISS index 建立完成")
    print(f"Index：{FAISS_INDEX_PATH}")
    print(f"Metadata：{FAISS_METADATA_PATH}")
    print(f"向量數量：{vectors.shape[0]}")
    print(f"向量維度：{dim}")


if __name__ == "__main__":
    main()
