from __future__ import annotations

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def build_text_encoder(
    vocab_size: int = 40000,
    seq_len: int = 200,
    embed_dim: int = 128,
    proj_dim: int = 128,
) -> keras.Model:
    """Text encoder that outputs an L2-normalized embedding vector."""
    text_in = keras.Input(shape=(), dtype=tf.string, name="text")

    vectorizer = layers.TextVectorization(
        max_tokens=vocab_size,
        output_mode="int",
        output_sequence_length=seq_len,
    )

    x = vectorizer(text_in)
    x = layers.Embedding(vocab_size, embed_dim, mask_zero=True)(x)
    x = layers.Bidirectional(layers.GRU(64, return_sequences=True))(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    emb = layers.Dense(proj_dim, activation=None, name="embedding")(x)

    # Normalize to make cosine similarity a dot-product
    emb = tf.nn.l2_normalize(emb, axis=-1)

    encoder = keras.Model(text_in, emb, name="text_encoder")
    encoder.vectorizer = vectorizer  # attach for easy .adapt()
    return encoder

def build_siamese(encoder: keras.Model) -> keras.Model:
    """Siamese similarity model: predicts whether two texts match."""
    a = keras.Input(shape=(), dtype=tf.string, name="a_text")
    b = keras.Input(shape=(), dtype=tf.string, name="b_text")

    ea = encoder(a)
    eb = encoder(b)

    # Since embeddings are normalized, dot == cosine similarity
    sim = layers.Dot(axes=1, normalize=False, name="cosine_dot")([ea, eb])

    # Feed similarity into a classifier head
    x = layers.Reshape((1,))(sim)
    out = layers.Dense(1, activation="sigmoid", name="match_prob")(x)

    model = keras.Model([a, b], out, name="siamese_recommender")
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=[keras.metrics.AUC(name="auc"), "accuracy"],
    )
    return model
