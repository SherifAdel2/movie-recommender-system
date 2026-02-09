from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from .utils import normalize_text

def load_movies(csv_path: str) -> pd.DataFrame:
    """Load and lightly clean the movies CSV.

    Expected columns (minimum): title, overview, genres
    Missing columns will be created as empty strings.
    """
    df = pd.read_csv(csv_path)

    for col in ["title", "overview", "genres"]:
        if col not in df.columns:
            df[col] = ""

    df["title"] = df["title"].fillna("").astype(str)
    df["overview"] = df["overview"].fillna("").astype(str)
    df["genres"] = df["genres"].fillna("").astype(str)

    # One unified text field for representation learning
    df["text"] = (
        df["title"].map(normalize_text)
        + " "
        + df["genres"].map(normalize_text)
        + " "
        + df["overview"].map(normalize_text)
    ).str.strip()

    # Drop tiny/empty rows
    df = df[df["text"].str.len() > 10].reset_index(drop=True)
    return df

def make_pairs(
    df: pd.DataFrame,
    n_neg: int = 3,
    seed: int = 42,
    test_size: float = 0.2,
):
    """Create (a_text, b_text, label) pairs.

    - Positive pair: movie text with itself (label=1)
    - Negative pairs: movie text with random other movie texts (label=0)

    This is a simple, robust training signal that works without user ratings.
    """
    rng = np.random.default_rng(seed)
    texts = df["text"].tolist()
    n = len(texts)

    a_texts, b_texts, labels = [], [], []

    for i in range(n):
        # Positive
        a_texts.append(texts[i])
        b_texts.append(texts[i])
        labels.append(1.0)

        # Negatives
        for _ in range(n_neg):
            j = int(rng.integers(0, n))
            while j == i:
                j = int(rng.integers(0, n))
            a_texts.append(texts[i])
            b_texts.append(texts[j])
            labels.append(0.0)

    a_texts = np.array(a_texts, dtype=object)
    b_texts = np.array(b_texts, dtype=object)
    labels = np.array(labels, dtype=np.float32)

    return train_test_split(
        a_texts,
        b_texts,
        labels,
        test_size=test_size,
        random_state=seed,
        stratify=labels,
    )
