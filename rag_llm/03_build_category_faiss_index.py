from pathlib import Path
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


MODEL_NAME = "intfloat/multilingual-e5-small"

KB_PATH = Path("processed/category_knowledge_base.json")

OUTPUT_DIR = Path("processed/category_faiss")
INDEX_PATH = OUTPUT_DIR / "category_rules.index"
METADATA_PATH = OUTPUT_DIR / "category_rules_metadata.json"
VECTORS_PATH = OUTPUT_DIR / "category_rule_vectors.npy"


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_category_text(item: dict) -> str:
    """
    將一個 category_knowledge_base item 轉成適合 embedding 的分類說明文字。
    """

    parts = [
        f"會辦單位：{item.get('unit', '')}",
        f"會辦類別：{item.get('category', '')}",
        f"label_name：{item.get('label_name', '')}",
        f"doc_type：{item.get('doc_type', '')}",
        f"定義：{item.get('definition', '')}",
        f"主要客戶意圖：{'；'.join(item.get('main_customer_intents', []))}",
        f"關鍵詞：{'；'.join(item.get('keywords', []))}",
        f"判斷規則：{'；'.join(item.get('decision_rules', []))}",
        f"不應歸類情況：{'；'.join(item.get('negative_rules', []))}",
        f"通話證據：{'；'.join(item.get('required_evidence_from_call', []))}",
        f"資料充足度：{item.get('data_sufficiency', '')}",
        f"限制：{item.get('limitations', '')}",
    ]

    return "\n".join(parts)


def normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return vectors / norms


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    kb = load_json(KB_PATH)

    if not isinstance(kb, list):
        raise ValueError("category_knowledge_base.json 應該是一個 list")

    model = SentenceTransformer(MODEL_NAME)

    category_texts = []
    metadata = []

    for idx, item in enumerate(kb):
        category_text = build_category_text(item)
        category_texts.append(f"passage: {category_text}")

        metadata.append({
            "category_index": idx,
            "label_name": item.get("label_name", ""),
            "doc_type": item.get("doc_type", ""),
            "unit": item.get("unit", ""),
            "category": item.get("category", ""),
            "definition": item.get("definition", ""),
            "num_examples": item.get("num_examples", 0),
            "data_sufficiency": item.get("data_sufficiency", ""),
            "category_text": category_text
        })

    print(f"分類數量：{len(category_texts)}")
    print("開始建立 category embeddings...")

    vectors = model.encode(
        category_texts,
        normalize_embeddings=True,
        show_progress_bar=True
    ).astype("float32")

    vectors = normalize_vectors(vectors).astype("float32")

    dim = vectors.shape[1]

    index = faiss.IndexFlatIP(dim)
    index.add(vectors)

    faiss.write_index(index, str(INDEX_PATH))
    np.save(VECTORS_PATH, vectors)

    with METADATA_PATH.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print("Category FAISS index 建立完成")
    print(f"Index：{INDEX_PATH}")
    print(f"Metadata：{METADATA_PATH}")
    print(f"Vectors：{VECTORS_PATH}")
    print(f"向量數量：{vectors.shape[0]}")
    print(f"向量維度：{vectors.shape[1]}")


if __name__ == "__main__":
    main()
