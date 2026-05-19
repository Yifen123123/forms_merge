import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity


NPZ_PATH = "processed/embeddings/call_embeddings.npz"
OUTPUT_PATH = "processed/top5_close_call_pairs.csv"
HIST_PATH = "processed/call_cosine_similarity_distribution.png"
TOP_K = 5


def normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return vectors / norms


def main():
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

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
    })

    vectors = normalize_vectors(vectors)

    sim_matrix = cosine_similarity(vectors)

    pairs = []

    for i in range(len(df)):
        for j in range(i + 1, len(df)):

            same_unit = df.loc[i, "unit"] == df.loc[j, "unit"]
            same_category = df.loc[i, "category"] == df.loc[j, "category"]

            pairs.append({
                "call_id_1": df.loc[i, "call_id"],
                "unit_1": df.loc[i, "unit"],
                "category_1": df.loc[i, "category"],
                "doc_type_1": df.loc[i, "doc_type"],

                "call_id_2": df.loc[j, "call_id"],
                "unit_2": df.loc[j, "unit"],
                "category_2": df.loc[j, "category"],
                "doc_type_2": df.loc[j, "doc_type"],

                "same_unit": same_unit,
                "same_category": same_category,

                "cosine_similarity": float(sim_matrix[i, j]),
                "cosine_distance": float(1 - sim_matrix[i, j]),
            })

    result = pd.DataFrame(pairs)

    result = result.sort_values(
        "cosine_similarity",
        ascending=False
    )

    top_k_result = result.head(TOP_K)

    top_k_result.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig"
    )

    print(f"Top {TOP_K} closest call pairs:")
    print(top_k_result)
    print(f"\nSaved to: {OUTPUT_PATH}")

    plt.figure(figsize=(10, 6))
    plt.hist(result["cosine_similarity"], bins=30, edgecolor="black")
    plt.xlabel("Cosine Similarity")
    plt.ylabel("Number of Call Pairs")
    plt.title("Distribution of Cosine Similarity Between Call Pairs")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(HIST_PATH, dpi=300)
    plt.show()

    print(f"Histogram saved to: {HIST_PATH}")


if __name__ == "__main__":
    main()
