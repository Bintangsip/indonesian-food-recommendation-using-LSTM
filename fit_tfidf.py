# fit_tfidf.py
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import os

# --- PATHS ---
PROCESSED_DATA_PATH = 'data/processed_recipes.csv'
MODELS_DIR = 'models'
TFIDF_VECTORIZER_PATH = os.path.join(MODELS_DIR, 'tfidf_vectorizer.pkl')
TFIDF_MATRIX_PATH = os.path.join(MODELS_DIR, 'tfidf_recipe_matrix.pkl')

if __name__ == "__main__":
    print("Memuat data yang sudah diproses...")
    if not os.path.exists(PROCESSED_DATA_PATH):
        print(f"✗ Error: File '{PROCESSED_DATA_PATH}' tidak ditemukan.")
    else:
        df = pd.read_csv(PROCESSED_DATA_PATH, delimiter=';')
        df.dropna(subset=['combined_features'], inplace=True)

        print("Memulai fitting TfidfVectorizer pada seluruh korpus...")
        tfidf_vectorizer = TfidfVectorizer()
        tfidf_vectorizer.fit(df['combined_features'])
        
        tfidf_recipe_matrix = tfidf_vectorizer.transform(df['combined_features'])

        os.makedirs(MODELS_DIR, exist_ok=True)

        with open(TFIDF_VECTORIZER_PATH, 'wb') as f:
            pickle.dump(tfidf_vectorizer, f)
        print(f"\n✅ TfidfVectorizer berhasil di-fit dan disimpan di: {TFIDF_VECTORIZER_PATH}")

        with open(TFIDF_MATRIX_PATH, 'wb') as f:
            pickle.dump(tfidf_recipe_matrix, f)
        print(f"✅ Matriks TF-IDF resep berhasil disimpan di: {TFIDF_MATRIX_PATH}")