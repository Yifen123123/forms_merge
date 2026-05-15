import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


NPZ_PATH = "processed/embeddings/call_embeddings.npz"
OUTPUT_PATH = "processed/top5_close_category_pairs.csv"


def normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return vectors / norms


def main():
    npz = np.load(NPZ_PATH, allow_pickle=True)

    vectors = npz["vectors"]
    categories = npz["categories"]
    units = npz["units"]
    doc_types = npz["doc_types"]
    call_ids = npz["call_ids"]

    df = pd.DataFrame({
        "call_id": call_ids,
        "unit": units,
        "doc_type": doc_types,
        "category": categories,
        "idx": range(len(categories)),
    })

    category_vectors = {}

    for category, group in df.groupby("category"):
        idxs = group["idx"].tolist()
        category_vectors[category] = vectors[idxs].mean(axis=0)

    category_names = list(category_vectors.keys())
    category_matrix = np.vstack([category_vectors[c] for c in category_names])
    category_matrix = normalize_vectors(category_matrix)

    sim_matrix = cosine_similarity(category_matrix)

    pairs = []

    for i in range(len(category_names)):
        for j in range(i + 1, len(category_names)):
            pairs.append({
                "category_1": category_names[i],
                "category_2": category_names[j],
                "cosine_similarity": float(sim_matrix[i, j]),
                "cosine_distance": float(1 - sim_matrix[i, j]),
            })

    result = pd.DataFrame(pairs)
    result = result.sort_values("cosine_similarity", ascending=False)

    top5 = result.head(5)
    top5.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    print("Top 5 closest category pairs:")
    print(top5)
    print(f"\nSaved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
