# food_recommendation_system.py
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import re
import time
import json
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import (
    cosine_similarity,
    euclidean_distances,
    manhattan_distances,
    pairwise_distances, # Untuk Jaccard
)
import pickle
import os
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import warnings
warnings.filterwarnings('ignore')


class FoodRecommendationSystem:
    def __init__(self, max_features=10000, max_sequence_length=512, embedding_dim=100, lstm_units=64,
                 learning_rate=0.001, batch_size=32, dropout=0.2, recurrent_dropout=0.2):
        self.max_features = max_features
        self.max_sequence_length = max_sequence_length
        self.embedding_dim = embedding_dim
        self.lstm_units = lstm_units
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.dropout = dropout
        self.recurrent_dropout = recurrent_dropout
        factory = StemmerFactory()
        self.stemmer = factory.create_stemmer()
        self.tokenizer = None
        self.label_encoder = None
        self.model = None
        self.all_data = None
        self.model = None
        self.all_data = None
        self.w2v_model = None
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.jaccard_vectorizer = None
        self.jaccard_matrix = None
        try:
            self.indo_stopwords = set(stopwords.words('indonesian'))
        except LookupError:
            print("Paket 'stopwords' NLTK tidak ditemukan, mengunduh...")
            nltk.download('stopwords')
            self.indo_stopwords = set(stopwords.words('indonesian'))
        self.indo_stopwords.update(['dan', 'dengan', 'secukupnya', 'hingga', 'sampai', 'lalu', 'buah', 'sdm', 'sdt', 'siung'])

    def clean_text(self, text):
        if isinstance(text, str):
            text = text.lower()
            text = re.sub(r'[^\w\s]', ' ', text)
            text = re.sub(r'\d+', ' ', text)
            text = re.sub(r'\s+', ' ', text).strip()
            return text
        return ""

    def process_text(self, text):
        """Fungsi ini sekarang melakukan stopword removal dan stemming."""
        if isinstance(text, str):
            try:
                tokens = word_tokenize(text)
            except LookupError:
                nltk.download('punkt')
                tokens = word_tokenize(text)
        
            # Hapus stopwords
            filtered_tokens = [token for token in tokens if token not in self.indo_stopwords]
        
            # Lakukan stemming
            stemmed_tokens = [self.stemmer.stem(token) for token in filtered_tokens]
        
            return ' '.join(stemmed_tokens)
        return ""

    def load_dataset(self, file_path, encoding='utf-8'):
        try:
            df = pd.read_csv(file_path, delimiter=';', encoding=encoding)
            print(f"Successfully loaded {file_path}")
            if 'category' not in df.columns: raise ValueError(f"Column 'category' not found in {file_path}")
            return df
        except FileNotFoundError:
            print(f"✗ Error: File tidak ditemukan di {file_path}")
            return pd.DataFrame()
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return pd.DataFrame()

    def preprocess_dataset(self, df):
        if df.empty: return df
        df.columns = df.columns.str.strip().str.lower()
        required_columns = ['title', 'ingredients', 'steps', 'category']
        if not all(col in df.columns for col in required_columns):
            missing = set(required_columns) - set(df.columns)
            print(f"Warning: Missing columns {list(missing)} in dataset")
            return pd.DataFrame()
    
        # Proses semua kolom teks dengan fungsi baru
        for col in ['title', 'ingredients', 'steps']:
            df[f'{col}_cleaned'] = df[col].apply(self.clean_text)
            df[f'{col}_processed'] = df[f'{col}_cleaned'].apply(self.process_text)
    
        # Pembobotan Judul
        title_with_weight = (df['title_processed'] + " ") * 2
        df['combined_features'] = title_with_weight + df['ingredients_processed'] + " " + df['steps_processed']
        return df

    def load_all_datasets(self, dataset_info):
            """
            Memuat, memproses, dan menggabungkan beberapa dataset dari daftar informasi file.
            """
            all_dfs = []
            print("Memulai pemuatan beberapa dataset...")
            for file_path, encoding in dataset_info:
                # Memuat satu dataset
                df = self.load_dataset(file_path, encoding=encoding)
            
            # Memproses dataset jika berhasil dimuat
                if not df.empty:
                    processed_df = self.preprocess_dataset(df)
                    if not processed_df.empty:
                        all_dfs.append(processed_df)

            if not all_dfs:
                print("✗ Peringatan: Tidak ada data yang berhasil dimuat. Mengembalikan DataFrame kosong.")
                return pd.DataFrame()
        
        # Gabungkan semua dataframe menjadi satu
            concatenated_df = pd.concat(all_dfs, ignore_index=True)
            self.all_data = concatenated_df.copy() # Simpan data untuk penggunaan nanti
            print(f"✅ Berhasil menggabungkan {len(all_dfs)} dataset.")
            return concatenated_df
    

# Di dalam kelas FoodRecommendationSystem
    def load_processed_data(self, file_path):
        """Memuat data yang sudah diproses dari file CSV."""
        try:
            self.all_data = pd.read_csv(file_path, delimiter=';')
            print(f"✅ Berhasil memuat data yang sudah diproses dari {file_path}")
        except Exception as e:
            print(f"✗ Gagal memuat data yang sudah diproses: {e}")
    def load_w2v_model(self, path):
        """Memuat model Word2Vec yang sudah dilatih."""
        try:
            self.w2v_model = Word2Vec.load(path)
            print(f"✅ Model Word2Vec berhasil dimuat dari {path}")
        except FileNotFoundError:
            print(f"✗ Peringatan: Model Word2Vec di '{path}' tidak ditemukan.")

    def load_tfidf_model(self, vectorizer_path, matrix_path):
        """Memuat TfidfVectorizer dan matriks resep yang sudah ada."""
        try:
            with open(vectorizer_path, 'rb') as f: self.tfidf_vectorizer = pickle.load(f)
            with open(matrix_path, 'rb') as f: self.tfidf_matrix = pickle.load(f)
            print("✅ Model TF-IDF (Vectorizer & Matrix) berhasil dimuat.")
        except FileNotFoundError:
            print("✗ Peringatan: File model TF-IDF tidak ditemukan.")

    def load_jaccard_model(self, vectorizer_path, matrix_path):
        """Memuat CountVectorizer (binary) dan matriks resep untuk Jaccard."""
        try:
            with open(vectorizer_path, 'rb') as f: self.jaccard_vectorizer = pickle.load(f)
            with open(matrix_path, 'rb') as f: self.jaccard_matrix = pickle.load(f)
            print("✅ Model Jaccard (Vectorizer & Matrix) berhasil dimuat.")
        except FileNotFoundError:
            print("✗ Peringatan: File model Jaccard tidak ditemukan.")

    def _get_document_vector(self, doc, model):
        """Fungsi helper untuk membuat vektor dokumen dengan merata-ratakan Word Vectors."""
        words = [word for word in doc.split() if word in model.wv]
        if not words: return np.zeros(model.vector_size)
        return np.mean(model.wv[words], axis=0)
    
    def feature_engineering(self, data):
        print("\nPerforming feature engineering...")
        self.tokenizer = Tokenizer(num_words=self.max_features, oov_token="<OOV>")
        self.tokenizer.fit_on_texts(data['combined_features'])
        sequences = self.tokenizer.texts_to_sequences(data['combined_features'])
        padded_sequences = pad_sequences(sequences, maxlen=self.max_sequence_length)
        self.label_encoder = LabelEncoder()
        data['category_encoded'] = self.label_encoder.fit_transform(data['category'])
        categories = data['category_encoded'].values
        return padded_sequences, categories

    def build_model(self, num_categories):
        print("\nBuilding LSTM model...")
        vocab_size = min(self.max_features, len(self.tokenizer.word_index) + 1)
       # Arsitektur LSTM
        model = Sequential([
            Embedding(input_dim=vocab_size, output_dim=self.embedding_dim, input_length=self.max_sequence_length),
            # Layer LSTM pertama harus memiliki return_sequences=True
            LSTM(units=self.lstm_units, 
                 dropout=self.dropout, 
                 recurrent_dropout=self.recurrent_dropout, 
                 return_sequences=True),
            # Layer LSTM kedua (terakhir) tidak perlu return_sequences=True
            LSTM(units=self.lstm_units // 2, 
                 dropout=self.dropout, 
                 recurrent_dropout=self.recurrent_dropout), # Bisa dengan unit lebih sedikit
            Dense(units=32, activation='relu'),
            Dropout(self.dropout),
            Dense(units=num_categories, activation='softmax')
        ])
        optimizer = Adam(learning_rate=self.learning_rate)
        model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        model.summary()
        self.model = model
        return model

    def train_model(self, X_train, y_train, epochs, patience, validation_split=0.2):
        print("\n--- Memulai Proses Training ---")
        early_stopping = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True, verbose=1)
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=self.batch_size,
            validation_split=validation_split,
            callbacks=[early_stopping],
            verbose=1
        )
        return history

    def evaluate_model(self, X_test, y_test):
        print("\n--- Mengevaluasi Model ---")
        y_pred_classes = np.argmax(self.model.predict(X_test), axis=1)
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred_classes),
            'precision': precision_score(y_test, y_pred_classes, average='weighted', zero_division=0),
            'recall': recall_score(y_test, y_pred_classes, average='weighted', zero_division=0),
            'f1_score': f1_score(y_test, y_pred_classes, average='weighted', zero_division=0)
        }
        print(f"\nAccuracy: {metrics['accuracy']:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred_classes, target_names=self.label_encoder.classes_, zero_division=0))

        print("\nConfusion Matrix:")
        cm = confusion_matrix(y_test, y_pred_classes)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=self.label_encoder.classes_, 
                    yticklabels=self.label_encoder.classes_)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Confusion Matrix')
        plt.show()
        
        return metrics, y_pred_classes

    def visualize_training_history(self, history):
        os.makedirs('visualizations', exist_ok=True)
        plt.figure(figsize=(15, 6))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Train Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy'); plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend(); plt.grid(True)
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss'); plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(); plt.grid(True)
        plt.tight_layout()
        plt.savefig('visualizations/model_performance.png')
        plt.show()

    def save_model(self, base_path_with_timestamp):
        print(f"\nMenyimpan model ke path: {base_path_with_timestamp}")
        os.makedirs(os.path.dirname(base_path_with_timestamp), exist_ok=True)
        self.model.save(f"{base_path_with_timestamp}_model.h5")
        with open(f"{base_path_with_timestamp}_tokenizer.pickle", 'wb') as f: pickle.dump(self.tokenizer, f)
        with open(f"{base_path_with_timestamp}_label_encoder.pickle", 'wb') as f: pickle.dump(self.label_encoder, f)
        print("✅ Model, Tokenizer, dan Label Encoder berhasil disimpan.")

    def load_model(self, base_path_with_timestamp):
        print(f"Loading model from base path: {base_path_with_timestamp}")
        self.model = load_model(f"{base_path_with_timestamp}_model.h5")
        with open(f"{base_path_with_timestamp}_tokenizer.pickle", 'rb') as f: self.tokenizer = pickle.load(f)
        with open(f"{base_path_with_timestamp}_label_encoder.pickle", 'rb') as f: self.label_encoder = pickle.load(f)
        print("✅ Model dan komponennya berhasil dimuat.")
    
    def get_recommendations(self, user_input, top_n=5, method='cosine'):
        if not all([self.tokenizer, self.model, self.label_encoder, self.all_data is not None]):
            raise RuntimeError("Sistem belum siap. Muat model dan data terlebih dahulu.")

        # Tahap 1: Prediksi Kategori
        cleaned_input = self.clean_text(user_input)
        cleaned_input_processed = self.process_text(cleaned_input)
        input_sequence = self.tokenizer.texts_to_sequences([cleaned_input_processed])
        padded_input = pad_sequences(input_sequence, maxlen=self.max_sequence_length)
        prediction = self.model.predict(padded_input)
        predicted_category_index = np.argmax(prediction[0])
        predicted_category = self.label_encoder.inverse_transform([predicted_category_index])[0]
        filtered_data = self.all_data[self.all_data['category'] == predicted_category].copy()
        if filtered_data.empty: filtered_data = self.all_data.copy()

        # Tahap 2 & 3: Vektorisasi & Perangkingan
        if method == 'cosine':
            if not all([self.tfidf_vectorizer, self.tfidf_matrix is not None]): raise RuntimeError("Model TF-IDF belum dimuat.")
            recipe_vectors = self.tfidf_matrix[filtered_data.index]
            user_vector = self.tfidf_vectorizer.transform([cleaned_input_processed])
            scores = cosine_similarity(user_vector, recipe_vectors)[0]
            top_indices = scores.argsort()[-top_n:][::-1]
    
        elif method == 'jaccard':
            if not all([self.jaccard_vectorizer, self.jaccard_matrix is not None]): raise RuntimeError("Model Jaccard belum dimuat.")
            recipe_vectors = self.jaccard_matrix[filtered_data.index]
            user_vector = self.jaccard_vectorizer.transform([cleaned_input_processed])
            user_vector_dense = user_vector.toarray()
            recipe_vectors_dense = recipe_vectors.toarray()
            distances = pairwise_distances(user_vector_dense, recipe_vectors_dense, metric='jaccard')[0]
            scores = 1 - distances
            top_indices = scores.argsort()[-top_n:][::-1]
    
        elif method in ['euclidean', 'manhattan', 'dot_product']:
            if self.w2v_model is None: raise RuntimeError("Model Word2Vec belum dimuat.")
            recipe_vectors = np.array([self._get_document_vector(doc, self.w2v_model) for doc in filtered_data['combined_features']])
            user_vector = self._get_document_vector(cleaned_input_processed, self.w2v_model)
            if method == 'euclidean':
                scores = euclidean_distances(user_vector.reshape(1, -1), recipe_vectors)[0]
                top_indices = scores.argsort()[:top_n]
            elif method == 'manhattan':
                scores = manhattan_distances(user_vector.reshape(1, -1), recipe_vectors)[0]
                top_indices = scores.argsort()[:top_n]
            elif method == 'dot_product':
                # normalisasi nilai dot product
                # 1. Normalisasi vektor input pengguna
                user_norm = np.linalg.norm(user_vector)
                # Hindari pembagian dengan nol jika vektornya kosong
                if user_norm == 0: user_norm = 1 
                normalized_user_vector = user_vector / user_norm

                # 2. Normalisasi semua vektor resep
                recipe_norms = np.linalg.norm(recipe_vectors, axis=1, keepdims=True)
                # Hindari pembagian dengan nol
                recipe_norms[recipe_norms == 0] = 1 
                normalized_recipe_vectors = recipe_vectors / recipe_norms

                # 3. Hitung dot product dengan vektor yang sudah dinormalisasi
                # Sekarang hasilnya akan setara dengan cosine similarity (0-1)
                scores = np.dot(normalized_recipe_vectors, normalized_user_vector.T).flatten()
                
                top_indices = scores.argsort()[-top_n:][::-1]
        else:
            raise ValueError(f"Metode '{method}' tidak valid.")

        # Tahap 4: Format Hasil
        recommendations = []
        for i, idx in enumerate(top_indices):
            original_index = filtered_data.index[idx]
            recipe = self.all_data.loc[original_index]
            score_value = scores[idx]
            display_value = 1 / (1 + score_value) if method in ['euclidean', 'manhattan'] else score_value
            recommendations.append({
                'title': recipe.get('title', ''), 'ingredients': recipe.get('ingredients', ''),
                'url': recipe.get('url', ''), 'similarity': display_value, 'category': recipe.get('category', '')
            })
        return recommendations

    def run_complete_workflow(self, dataset_info, epochs, patience):
        start_time = time.time()
        all_data = self.load_all_datasets(dataset_info)
        
        if all_data.empty:
            print("Proses dihentikan karena tidak ada data yang berhasil dimuat.")
            return self

        X, y = self.feature_engineering(all_data)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        self.build_model(len(self.label_encoder.classes_))
        history = self.train_model(X_train, y_train, epochs=epochs, patience=patience)
        metrics, y_pred_classes = self.evaluate_model(X_test, y_test)
        
        training_timestamp = int(start_time)
        MODELS_DIR = 'models'
        TRAINING_LOGS_DIR = 'training_logs'
        os.makedirs(TRAINING_LOGS_DIR, exist_ok=True)
        model_base_path = os.path.join(MODELS_DIR, f'model_{training_timestamp}')
        self.save_model(model_base_path)

        training_duration = time.time() - start_time
        cm = confusion_matrix(y_test, y_pred_classes)
        
        log_entry = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time)),
            "model_files": {"base_path": model_base_path},
            "parameters": {
                "max_features": self.max_features, "max_sequence_length": self.max_sequence_length,
                "embedding_dim": self.embedding_dim, "lstm_units": self.lstm_units,
                "learning_rate": self.learning_rate, "batch_size": self.batch_size,
                "dropout": self.dropout, "recurrent_dropout": self.recurrent_dropout,
                "epochs": epochs, "patience": patience
            },
            "metrics": metrics,
            "training_duration_seconds": round(training_duration, 2),
            "training_history": history.history,
            "confusion_matrix": cm.tolist()
        }
        
        log_filename = os.path.join(TRAINING_LOGS_DIR, f"training_log_{training_timestamp}.json")
        with open(log_filename, 'w') as f: json.dump(log_entry, f, indent=4)
        print(f"✅ Log pelatihan berhasil disimpan di {log_filename}")

        self.visualize_training_history(history)
        print("\n✅ Food recommendation system training complete!")
        return self