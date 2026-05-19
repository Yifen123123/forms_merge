from pathlib import Path
import json
import numpy as np
import ollama


DATASET_PATH = Path("processed/classifier_dataset.json")
CALLS_DIR = Path("calls")
OUTPUT_DIR = Path("processed/embeddings")

MODEL_NAME = "nomic-embed-text"

EMBEDDINGS_PATH = OUTPUT_DIR / "call_embeddings_nomic_no_chunk.npz"
METADATA_PATH = OUTPUT_DIR / "embedding_metadata_nomic_no_chunk.json"


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


def normalize_vector(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)

    if norm == 0:
        return vector

    return vector / norm


def embed_text_with_ollama(text: str) -> np.ndarray:
    response = ollama.embed(
        model=MODEL_NAME,
        input=text
    )

    vector = np.array(response["embeddings"][0], dtype=np.float32)
    vector = normalize_vector(vector)

    return vector.astype("float32")


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

        print(f"[{idx}/{len(dataset)}] embedding call_id={call_id}")

        text = read_call_text(call_id)

        call_vector = embed_text_with_ollama(text)

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
            "chunk_method": "no_chunk_full_text",
            "num_chunks": 1,
            "text_length_chars": len(text),
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

    print("\nNomic embedding 建立完成")
    print(f"向量檔案：{EMBEDDINGS_PATH}")
    print(f"metadata：{METADATA_PATH}")
    print(f"vectors shape: {vectors.shape}")


if __name__ == "__main__":
    main()
