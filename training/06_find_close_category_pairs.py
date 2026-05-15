# 06_find_close_unit_category_pairs.py

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


NPZ_PATH = "processed/embeddings/call_embeddings.npz"
OUTPUT_PATH = "processed/top5_close_unit_category_pairs.csv"
TOP_K = 5


def normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return vectors / norms


def main():
    npz = np.load(NPZ_PATH, allow_pickle=True)

    vectors = npz["vectors"]
    call_ids = npz["call_ids"]
    units = npz["units"]
    categories = npz["categories"]
    doc_types = npz["doc_types"]

    df = pd.DataFrame({
        "call_id": call_ids,
        "unit": units,
        "category": categories,
        "doc_type": doc_types,
        "idx": range(len(categories)),
    })

    # 用 (unit, category) 當作一個類別 identity
    prototypes = []

    for (unit, category), group in df.groupby(["unit", "category"]):
        idxs = group["idx"].tolist()
        proto_vector = vectors[idxs].mean(axis=0)

        prototypes.append({
            "unit": unit,
            "category": category,
            "sample_count": len(idxs),
            "vector": proto_vector,
        })

    proto_df = pd.DataFrame(prototypes)

    proto_vectors = np.vstack(proto_df["vector"].to_numpy())
    proto_vectors = normalize_vectors(proto_vectors)

    sim_matrix = cosine_similarity(proto_vectors)

    pairs = []

    for i in range(len(proto_df)):
        for j in range(i + 1, len(proto_df)):
            pairs.append({
                "unit_1": proto_df.loc[i, "unit"],
                "category_1": proto_df.loc[i, "category"],
                "sample_count_1": int(proto_df.loc[i, "sample_count"]),

                "unit_2": proto_df.loc[j, "unit"],
                "category_2": proto_df.loc[j, "category"],
                "sample_count_2": int(proto_df.loc[j, "sample_count"]),

                "cosine_similarity": float(sim_matrix[i, j]),
                "cosine_distance": float(1 - sim_matrix[i, j]),
            })

    result = pd.DataFrame(pairs)
    result = result.sort_values("cosine_similarity", ascending=False)

    top_k_result = result.head(TOP_K)

    top_k_result.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig"
    )

    print(f"Top {TOP_K} closest (unit, category) pairs:")
    print(top_k_result)
    print(f"\nSaved to: {OUTPUT_PATH}")

    plt.figure(figsize=(10, 6))
    plt.hist(result["cosine_similarity"], bins=30, edgecolor="black")
    plt.xlabel("Cosine Similarity")
    plt.ylabel("Number of Category Pairs")
    plt.title("Distribution of Cosine Similarity Between Category Pairs")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("processed/cosine_similarity_distribution.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    main()
