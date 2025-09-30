# 🎬 Movie Recommender System (Content-Based with Neural Networks)

This project implements a **content-based recommender system** using **Python, TensorFlow/Keras, and embeddings**.  
It is inspired by Andrew Ng’s _Machine Learning Specialization (Course 3, Week 2: Recommender Systems)_ on Coursera.

---

## 📌 Project Overview

- Build a **movie recommendation engine** that predicts user ratings based on embeddings for users and movies.
- Demonstrates the use of **neural networks for content-based filtering**.
- Uses a small **synthetic dataset** of users, movies, and ratings for demonstration purposes.

---

## 🛠️ Technologies Used

- **Python 3**
- **NumPy, pandas** → data preprocessing
- **TensorFlow / Keras** → model building
- **Matplotlib** → optional visualization
- **Jupyter Notebook**

---

## 📂 Files

- `movie_recommender.ipynb` → Main Jupyter Notebook with the full implementation.
- `README.md` → Project documentation.

---

## 🚀 How It Works

1. **Dataset Creation**

   - Small synthetic dataset of users, movies, and ratings.
   - Encodes users and movies into categorical indices.

2. **Neural Network with Embeddings**

   - User embedding layer → learns vector representation for users.
   - Movie embedding layer → learns vector representation for movies.
   - Concatenates both embeddings and passes them through dense layers.
   - Trains to predict ratings using Mean Squared Error (MSE).

3. **Recommendation Function**
   - For a given user, predicts ratings for all movies.
   - Ranks movies and returns the **Top-N recommendations**.

---
