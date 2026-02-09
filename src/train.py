from __future__ import annotations

import os
import numpy as np
from tensorflow import keras

from .data import load_movies, make_pairs
from .model import build_text_encoder, build_siamese

def train(
    csv_path: str,
    out_dir: str = "models",
    n_neg: int = 3,
    batch_size: int = 256,
    epochs: int = 15,
    seed: int = 42,
) -> dict:
    os.makedirs(out_dir, exist_ok=True)

    df = load_movies(csv_path)

    encoder = build_text_encoder()
    encoder.vectorizer.adapt(df["text"].values)

    model = build_siamese(encoder)

    a_train, a_val, b_train, b_val, y_train, y_val = make_pairs(df, n_neg=n_neg, seed=seed)

    callbacks = [
        keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
        keras.callbacks.ModelCheckpoint(os.path.join(out_dir, "siamese.keras"), save_best_only=True),
    ]

    history = model.fit(
        {"a_text": a_train, "b_text": b_train},
        y_train,
        validation_data=({"a_text": a_val, "b_text": b_val}, y_val),
        batch_size=batch_size,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1,
    )

    # Export embeddings for fast retrieval
    text_encoder = model.get_layer("text_encoder")
    movie_embs = text_encoder.predict(df["text"].values, batch_size=batch_size, verbose=1)

    np.save(os.path.join(out_dir, "movie_embeddings.npy"), movie_embs)
    df.to_csv(os.path.join(out_dir, "movies_clean.csv"), index=False)

    return {
        "num_movies": int(len(df)),
        "out_dir": out_dir,
        "best_val_auc": float(max(history.history.get("val_auc", [0.0]))),
        "best_val_acc": float(max(history.history.get("val_accuracy", [0.0]))),
    }
