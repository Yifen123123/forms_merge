import json
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# 讀資料
with open("processed/classifier_dataset.json", "r", encoding="utf-8") as f:
    dataset = json.load(f)

embeddings = np.load("processed/embeddings.npy")

# 取得 label
labels = [item["category"] for item in dataset]

# 如果同一類有多筆，取平均作為 category prototype
df = pd.DataFrame({
    "label": labels,
    "idx": range(len(labels))
})

category_vectors = {}

for label, group in df.groupby("label"):
    idxs = group["idx"].tolist()
    category_vectors[label] = embeddings[idxs].mean(axis=0)

categories = list(category_vectors.keys())
vectors = np.vstack([category_vectors[c] for c in categories])

# normalize
vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

# 計算 cosine similarity
sim_matrix = cosine_similarity(vectors)

pairs = []

for i in range(len(categories)):
    for j in range(i + 1, len(categories)):
        pairs.append({
            "category_1": categories[i],
            "category_2": categories[j],
            "cosine_similarity": sim_matrix[i, j],
            "cosine_distance": 1 - sim_matrix[i, j],
        })

result = pd.DataFrame(pairs)
result = result.sort_values("cosine_similarity", ascending=False)

# 取 top 5 最接近
top5 = result.head(5)

top5.to_csv("processed/top5_close_category_pairs.csv", index=False, encoding="utf-8-sig")

print(top5)
