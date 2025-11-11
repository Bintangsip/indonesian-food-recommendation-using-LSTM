# fit_jaccard.py
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import pickle
import os

# --- PATHS ---
PROCESSED_DATA_PATH = 'data/processed_recipes.csv'
MODELS_DIR = 'models'
JACCARD_VECTORIZER_PATH = os.path.join(MODELS_DIR, 'jaccard_vectorizer.pkl')
JACCARD_MATRIX_PATH = os.path.join(MODELS_DIR, 'jaccard_recipe_matrix.pkl')

if __name__ == "__main__":
    print("Memuat data yang sudah diproses...")
    if not os.path.exists(PROCESSED_DATA_PATH):
        print(f"✗ Error: File '{PROCESSED_DATA_PATH}' tidak ditemukan.")
    else:
        df = pd.read_csv(PROCESSED_DATA_PATH, delimiter=';')
        df.dropna(subset=['combined_features'], inplace=True)

        print("Memulai fitting CountVectorizer (binary) untuk Jaccard...")
        jaccard_vectorizer = CountVectorizer(binary=True)
        jaccard_vectorizer.fit(df['combined_features'])
        jaccard_recipe_matrix = jaccard_vectorizer.transform(df['combined_features'])

        os.makedirs(MODELS_DIR, exist_ok=True)

        with open(JACCARD_VECTORIZER_PATH, 'wb') as f:
            pickle.dump(jaccard_vectorizer, f)
        print(f"\n✅ Jaccard Vectorizer berhasil disimpan di: {JACCARD_VECTORIZER_PATH}")

        with open(JACCARD_MATRIX_PATH, 'wb') as f:
            pickle.dump(jaccard_recipe_matrix, f)
        print(f"✅ Matriks Resep Jaccard berhasil disimpan di: {JACCARD_MATRIX_PATH}")