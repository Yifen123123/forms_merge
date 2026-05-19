import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity


NPZ_PATH = "processed/embeddings/call_embeddings_nomic_no_chunk.npz"

OUTPUT_PATH = "processed/all_close_call_pairs_nomic.csv"
HIST_PATH = "processed/call_cosine_similarity_distribution_nomic.png"


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

            cosine_sim = float(sim_matrix[i, j])

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

                "cosine_similarity": cosine_sim,
                "cosine_distance": float(1 - cosine_sim),
            })

    result = pd.DataFrame(pairs)

    result = result.sort_values(
        "cosine_similarity",
        ascending=False
    )

    result.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig"
    )

    print("Pairwise cosine similarity 計算完成")
    print(f"資料筆數：{len(df)}")
    print(f"pair 數量：{len(result)}")
    print(f"輸出檔案：{OUTPUT_PATH}")
    print()
    print("cosine_similarity 統計：")
    print(result["cosine_similarity"].describe())

    print()
    print("Top 5 最相似：")
    print(result.head(5))

    print()
    print("Bottom 5 最不相似：")
    print(result.tail(5))

    plt.figure(figsize=(10, 6))
    plt.hist(
        result["cosine_similarity"],
        bins=30,
        edgecolor="black"
    )
    plt.xlabel("Cosine Similarity")
    plt.ylabel("Number of Call Pairs")
    plt.title("Distribution of Cosine Similarity Between Call Pairs - Nomic Embed")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(HIST_PATH, dpi=300)
    plt.show()

    print(f"Histogram saved to: {HIST_PATH}")


if __name__ == "__main__":
    main()
