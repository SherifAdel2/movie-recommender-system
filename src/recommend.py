from __future__ import annotations

import numpy as np
import pandas as pd
from tensorflow import keras

def cosine_topk(query_vec: np.ndarray, matrix: np.ndarray, k: int = 10):
    scores = matrix @ query_vec  # dot on normalized vectors
    idx = np.argsort(-scores)[:k]
    return idx, scores[idx]

def recommend_by_title(
    title: str,
    k: int = 10,
    models_dir: str = "models",
) -> pd.DataFrame | None:
    df = pd.read_csv(f"{models_dir}/movies_clean.csv")
    embs = np.load(f"{models_dir}/movie_embeddings.npy")

    # Load model just to ensure same preprocessing is available if needed later
    _ = keras.models.load_model(f"{models_dir}/siamese.keras", compile=False)

    # Find movie index
    exact = df[df["title"].str.lower() == title.lower()]
    if len(exact) == 0:
        partial = df[df["title"].str.lower().str.contains(title.lower())]
        if len(partial) == 0:
            return None
        i = int(partial.index[0])
    else:
        i = int(exact.index[0])

    q = embs[i]
    idx, scores = cosine_topk(q, embs, k=k + 1)  # +1 to include itself
    idx = [j for j in idx if j != i][:k]

    recs = df.iloc[idx][["title", "genres"]].copy()
    recs["score"] = [float(embs[j] @ q) for j in idx]
    recs.insert(0, "query_title", df.loc[i, "title"])
    return recs.sort_values("score", ascending=False).reset_index(drop=True)
