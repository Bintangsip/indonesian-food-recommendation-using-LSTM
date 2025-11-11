# indonesian food recommendation using LSTM

-----

# 🍳 Sistem Rekomendasi Makanan Cerdas (Foodie Finder)

Ini adalah proyek sistem rekomendasi makanan cerdas yang dibangun menggunakan pendekatan *hybrid* (campuran). Sistem ini menggabungkan model Deep Learning (LSTM) untuk klasifikasi kategori dengan berbagai metode *content-based filtering* (TF-IDF, Jaccard, Word2Vec) untuk perangkingan resep.

Aplikasi ini dilengkapi dengan antarmuka web yang dibangun menggunakan Streamlit, memungkinkan pengguna untuk:

  * Mencari rekomendasi resep berdasarkan preferensi (input teks).
  * Memilih model LSTM yang telah dilatih.
  * Memilih metode kalkulasi kemiripan (ranking) yang berbeda.
  * Melihat log pelatihan dan performa dari setiap model yang dilatih.

## 🌟 Fitur Utama

  * **Model Hybrid:** Menggunakan LSTM untuk memprediksi kategori (Klasifikasi) dan metrik similarity untuk mencari resep (Ranking).
  * **Dukungan Multi-Metrik:** Mendukung perangkingan resep menggunakan **Cosine Similarity** (TF-IDF), **Jaccard Similarity** (CountVectorizer), dan **Euclidean/Manhattan/Dot Product** (Word2Vec).
  * **Antarmuka Streamlit:** UI yang bersih dan interaktif untuk mencari resep dan menganalisis model.
  * **Analisis Pelatihan:** Halaman "Log Pelatihan" secara otomatis memuat, menganalisis, dan memvisualisasikan performa model (termasuk Confusion Matrix dan grafik) dari file log JSON.
  * **Pipeline Data Lengkap:** Skrip terpisah untuk pembersihan data, *feature engineering* (TF-IDF, W2V, Jaccard), dan pelatihan model LSTM.
  * **Pemrosesan Teks (NLP):** Menggunakan NLTK dan Sastrawi untuk *stopwords removal* dan *stemming* Bahasa Indonesia.

## 💡 Cara Kerja Sistem

Sistem ini bekerja dalam dua tahap utama untuk memberikan rekomendasi:

1.  **Tahap 1: Klasifikasi Kategori (LSTM)**

      * Saat pengguna memasukkan *query* (misal: "resep ayam pedas manis"), *query* tersebut diproses.
      * Model *Stacked LSTM* yang telah dilatih memprediksi kategori yang paling relevan untuk *query* tersebut (misal: "Masakan Utama").

2.  **Tahap 2: Perangkingan Kemiripan (Content-Based)**

      * Sistem memfilter database dan hanya mengambil resep dari kategori yang diprediksi ("Masakan Utama").
      * *Query* pengguna kemudian dibandingkan dengan semua resep dalam kategori tersebut menggunakan metode yang dipilih di UI:
          * **Cosine:** Menghitung kemiripan berdasarkan vektor **TF-IDF**.
          * **Jaccard:** Menghitung kemiripan berdasarkan kata-kata yang sama (set).
          * **Euclidean/Lainnya:** Menghitung jarak berdasarkan vektor **Word2Vec**.
      * 5 resep dengan skor kemiripan tertinggi akan ditampilkan sebagai rekomendasi.

## 📁 Struktur Proyek

```
.
├── data/
│   ├── (Letakkan file .csv mentah Anda di sini)
│   └── processed_recipes.csv     # Dihasilkan oleh preprocess_data.py
├── models/
│   ├── tfidf_vectorizer.pkl      # Dihasilkan oleh fit_tfidf.py
│   ├── tfidf_recipe_matrix.pkl   # Dihasilkan oleh fit_tfidf.py
│   ├── jaccard_vectorizer.pkl    # Dihasilkan oleh fit_jaccard.py
│   ├── jaccard_recipe_matrix.pkl # Dihasilkan oleh fit_jaccard.py
│   ├── word2vec_recipes.model    # Dihasilkan oleh train_word2vec.py
│   └── ... (model LSTM, tokenizer, dll. dari train_model.py)
├── training_logs/
│   └── training_log_xxxx.json    # Dihasilkan oleh train_model.py
├── food_recommendation_system.py # File kelas inti
├── preprocess_data.py            # [RUN 1] Skrip untuk membersihkan data
├── fit_tfidf.py                  # [RUN 2a] Skrip untuk membuat model TF-IDF
├── fit_jaccard.py                # [RUN 2b] Skrip untuk membuat model Jaccard
├── train_word2vec.py             # [RUN 2c] Skrip untuk melatih Word2Vec
├── train_model.py                # [RUN 3] Skrip untuk melatih model LSTM
├── streamlit_ui.py               # [RUN 4] Skrip untuk menjalankan aplikasi UI
├── requirements.txt              # (Anda perlu membuat file ini)
└── README.md                     # File ini
```

## 🚀 Instalasi

1.  **Clone Repositori**

    ```bash
    git clone https://github.com/username/repository-name.git
    cd repository-name
    ```

2.  **Buat Virtual Environment** (Disarankan)

    ```bash
    python -m venv venv
    source venv/bin/activate  # Di Windows: venv\Scripts\activate
    ```

3.  **Install Dependensi**
    Anda perlu menginstal semua library yang dibutuhkan. Sebaiknya buat file `requirements.txt` dari impor yang ada.

    ```bash
    pip install pandas numpy scikit-learn tensorflow streamlit gensim nltk sastrawi seaborn matplotlib
    ```

4.  **Download Data NLTK**
    Jalankan interpreter Python dan unduh paket NLTK yang diperlukan:

    ```python
    import nltk
    nltk.download('stopwords')
    nltk.download('punkt')
    ```

## 🛠️ Alur Penggunaan (Cara Menjalankan)

Ikuti langkah-langkah ini secara berurutan untuk memproses data dan melatih model Anda dari awal.

### Langkah 0: Siapkan Data Mentah

Tempatkan semua file `.csv` dataset resep Anda ke dalam direktori `/data`. Pastikan file-file ini memiliki kolom `title`, `ingredients`, `steps`, dan `category`.

### Langkah 1: Pra-pemrosesan Data

Skrip ini akan memuat semua CSV di `/data`, membersihkannya, melakukan stemming/stopword removal, dan menyimpannya sebagai satu file `processed_recipes.csv`.

```bash
python preprocess_data.py
```

### Langkah 2: Latih Model Vektorisasi (Similarity)

Jalankan ketiga skrip ini untuk membuat dan menyimpan model TF-IDF, Jaccard, dan Word2Vec. Model-model ini diperlukan untuk tahap perangkingan di UI.

```bash
python fit_tfidf.py
python fit_jaccard.py
python train_word2vec.py
```

### Langkah 3: Latih Model LSTM (Klasifikasi)

Skrip ini akan memuat data yang sudah diproses, melakukan *undersampling* untuk menyeimbangkan data, lalu melatih model LSTM. Model, tokenizer, dan log pelatihan akan disimpan di direktori `/models` dan `/training_logs`.

```bash
python train_model.py
```

Anda dapat mengulangi langkah ini beberapa kali (mungkin dengan mengubah hiperparameter di `train_model.py`) untuk menghasilkan beberapa versi model.

### Langkah 4: Jalankan Aplikasi Streamlit

Setelah semua model (LSTM, TF-IDF, Jaccard, W2V) dan data (`processed_recipes.csv`) ada di tempatnya, jalankan aplikasi UI:

```bash
streamlit run streamlit_ui.py
```

Buka browser Anda dan akses alamat (biasanya `http://localhost:8501`) untuk mulai menggunakan sistem rekomendasi.
