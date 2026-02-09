# 🎬 Movie Recommender System (Content-Based) — Python + Keras

A **content-based** movie recommendation system that learns **neural embeddings** from movie metadata
(title + genres + overview) and returns **Top‑K similar movies** using **cosine similarity**.


## ✨ What’s inside

- **Neural feature representations** using a text encoder (Keras)
- **Siamese training** with positive/negative pairs (contrastive-style classification)
- **Embedding export** for fast retrieval
- **Cosine similarity search** for Top‑K recommendations
- **Evaluation script** (retrieval @K on a simple proxy task)

## 📦 Project structure

```
movie-recommender/
  src/
    data.py          # load/clean data + pair generation
    model.py         # text encoder + siamese network
    train.py         # training pipeline + export embeddings
    recommend.py     # cosine search + recommend by title
    evaluate.py      # simple retrieval evaluation
    cli.py           # Typer CLI entrypoint
    utils.py
  data/              # (not committed) put your CSV here
  models/            # saved model + embeddings
  outputs/           # evaluation results
```

## ✅ Requirements

- Python 3.9+
- TensorFlow/Keras

Install:

```bash
pip install -r requirements.txt
```

## 📁 Dataset format

Put a CSV file at: `data/movies.csv`

Minimum columns (case-sensitive):

- `title`
- `overview` (or plot/description)
- `genres` (string; can be pipe-separated or JSON-ish — we keep it as text)

Example header:

```csv
title,overview,genres
```

> If your dataset uses different column names, edit `src/data.py` mapping.

## 🚀 Train & export embeddings

```bash
python -m src.cli train --csv data/movies.csv
```

This will create:

- `models/siamese.keras`
- `models/movie_embeddings.npy`
- `models/movies_clean.csv`

## 🔎 Recommend similar movies

```bash
python -m src.cli recommend --title "Avatar" --k 10
```

## 📊 Evaluate (simple retrieval proxy)

```bash
python -m src.cli evaluate --k 10
```

The evaluation is a lightweight proxy: it checks whether the nearest neighbor retrieval behaves sensibly
for self-similarity and some synthetic perturbations.

