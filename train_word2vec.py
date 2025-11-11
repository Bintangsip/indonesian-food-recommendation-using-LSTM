# train_word2vec.py
import pandas as pd
from gensim.models import Word2Vec
import os

# --- PATHS ---
PROCESSED_DATA_PATH = 'data/processed_recipes.csv'
MODELS_DIR = 'models'
W2V_MODEL_PATH = os.path.join(MODELS_DIR, 'word2vec_recipes.model')

# --- KONFIGURASI MODEL W2V ---
VECTOR_SIZE = 100 # Dimensi vektor kata
WINDOW = 5        # Jarak maksimum antara kata target dan konteksnya
MIN_COUNT = 2     # Abaikan kata yang muncul kurang dari ini
WORKERS = 4       # Jumlah thread CPU yang digunakan

if __name__ == "__main__":
    print("Memuat data yang sudah diproses...")
    if not os.path.exists(PROCESSED_DATA_PATH):
        print(f"✗ Error: File '{PROCESSED_DATA_PATH}' tidak ditemukan. Jalankan preprocess_data.py terlebih dahulu.")
    else:
        df = pd.read_csv(PROCESSED_DATA_PATH, delimiter=';')
        
        # Pastikan kolom 'combined_features' tidak kosong
        df.dropna(subset=['combined_features'], inplace=True)

        print("Mempersiapkan korpus untuk training Word2Vec...")
        # Ubah setiap baris teks menjadi daftar kata (token)
        tokenized_corpus = [doc.split() for doc in df['combined_features']]
        
        print(f"Memulai training model Word2Vec dengan {len(tokenized_corpus)} resep...")
        model = Word2Vec(sentences=tokenized_corpus,
                         vector_size=VECTOR_SIZE,
                         window=WINDOW,
                         min_count=MIN_COUNT,
                         workers=WORKERS)
        
        os.makedirs(MODELS_DIR, exist_ok=True)
        
        model.save(W2V_MODEL_PATH)
        
        print(f"\n✅ Model Word2Vec berhasil dilatih dan disimpan di: {W2V_MODEL_PATH}")
        print(f"Total vocabulary: {len(model.wv.index_to_key)} kata.")