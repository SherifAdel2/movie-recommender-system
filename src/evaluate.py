from __future__ import annotations

import os
import numpy as np
import pandas as pd

def retrieval_self_check(models_dir: str = "models", k: int = 10) -> dict:
    """A lightweight sanity-check evaluation.

    Since we don't have user interactions/ratings in a pure content-based setup,
    we evaluate retrieval behavior with a proxy:
    - Nearest neighbor of each item should be itself (rank-1)
    - Compute average self-rank within top-K (should be 1)
    """
    embs = np.load(f"{models_dir}/movie_embeddings.npy")
    # dot on normalized vectors
    sim = embs @ embs.T

    # rank positions (descending)
    # self similarity should be max (rank 1)
    ranks = []
    for i in range(sim.shape[0]):
        order = np.argsort(-sim[i])
        rank = int(np.where(order == i)[0][0]) + 1  # 1-based
        ranks.append(rank)

    ranks = np.array(ranks)
    topk_acc = float(np.mean(ranks <= k))
    mean_rank = float(np.mean(ranks))

    return {"k": k, "topk_self_accuracy": topk_acc, "mean_self_rank": mean_rank, "n": int(len(ranks))}

def save_report(report: dict, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    pd.DataFrame([report]).to_csv(out_path, index=False)
