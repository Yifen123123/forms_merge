from pathlib import Path
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


# =========================
# 基本設定
# =========================

MODEL_NAME = "intfloat/multilingual-e5-small"

KB_PATH = Path("processed/category_knowledge_base.json")

OUTPUT_DIR = Path("processed/category_faiss")
INDEX_PATH = OUTPUT_DIR / "category_rules.index"
METADATA_PATH = OUTPUT_DIR / "category_rules_metadata.json"
VECTORS_PATH = OUTPUT_DIR / "category_rule_vectors.npy"
CONFIG_PATH = OUTPUT_DIR / "category_faiss_config.json"


# =========================
# 工具函式
# =========================

def load_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"找不到檔案：{path}")

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def safe_join_list(value) -> str:
    """
    避免 LLM 產出的欄位不是 list 時造成錯誤。
    """
    if isinstance(value, list):
        return "；".join(str(x) for x in value)

    if isinstance(value, str):
        return value

    return ""


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
        f"主要客戶意圖：{safe_join_list(item.get('main_customer_intents', []))}",
        f"關鍵詞：{safe_join_list(item.get('keywords', []))}",
        f"判斷規則：{safe_join_list(item.get('decision_rules', []))}",
        f"不應歸類情況：{safe_join_list(item.get('negative_rules', []))}",
        f"通話證據：{safe_join_list(item.get('required_evidence_from_call', []))}",
        f"可能混淆類別：{safe_join_list(item.get('possible_confusing_categories', []))}",
        f"資料充足度：{item.get('data_sufficiency', '')}",
        f"限制：{item.get('limitations', '')}",
    ]

    return "\n".join(parts)


def build_metadata(kb: list[dict], category_texts: list[str]) -> list[dict]:
    metadata = []

    for idx, item in enumerate(kb):
        metadata.append({
            "category_index": idx,
            "label_name": item.get("label_name", ""),
            "doc_type": item.get("doc_type", ""),
            "unit": item.get("unit", ""),
            "category": item.get("category", ""),
            "definition": item.get("definition", ""),
            "num_examples": item.get("num_examples", 0),
            "data_sufficiency": item.get("data_sufficiency", ""),
            "example_call_ids": item.get("example_call_ids", []),
            "category_text": category_texts[idx],
        })

    return metadata


# =========================
# 主程式
# =========================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    kb = load_json(KB_PATH)

    if not isinstance(kb, list):
        raise ValueError("category_knowledge_base.json 應該是一個 list")

    if len(kb) == 0:
        raise ValueError("category_knowledge_base.json 是空的，無法建立 FAISS index")

    category_texts = []

    for item in kb:
        category_text = build_category_text(item)

        # E5 系列建議 passage/query prefix
        category_texts.append(f"passage: {category_text}")

    metadata = build_metadata(kb, category_texts)

    print(f"分類數量：{len(category_texts)}")
    print(f"使用 embedding model：{MODEL_NAME}")
    print("開始建立 category embeddings...")

    model = SentenceTransformer(MODEL_NAME)

    vectors = model.encode(
        category_texts,
        normalize_embeddings=True,
        show_progress_bar=True,
        batch_size=16,
    ).astype("float32")

    dim = vectors.shape[1]

    index = faiss.IndexFlatIP(dim)
    index.add(vectors)

    faiss.write_index(index, str(INDEX_PATH))
    np.save(VECTORS_PATH, vectors)

    save_json(metadata, METADATA_PATH)

    config = {
        "model_name": MODEL_NAME,
        "kb_path": str(KB_PATH),
        "index_path": str(INDEX_PATH),
        "metadata_path": str(METADATA_PATH),
        "vectors_path": str(VECTORS_PATH),
        "total_categories": len(category_texts),
        "embedding_dim": dim,
        "faiss_index_type": "IndexFlatIP",
        "normalized_embeddings": True,
        "similarity": "cosine_similarity_via_inner_product",
    }

    save_json(config, CONFIG_PATH)

    print("Category FAISS index 建立完成")
    print(f"Index：{INDEX_PATH}")
    print(f"Metadata：{METADATA_PATH}")
    print(f"Vectors：{VECTORS_PATH}")
    print(f"Config：{CONFIG_PATH}")
    print(f"向量數量：{vectors.shape[0]}")
    print(f"向量維度：{vectors.shape[1]}")


if __name__ == "__main__":
    main()
