import os
import pandas as pd
from sklearn.model_selection import train_test_split
from food_recommendation_system import FoodRecommendationSystem
import time
import json
from sklearn.metrics import confusion_matrix
from sklearn.utils import resample

# --- PATHS ---
PROCESSED_DATA_PATH = 'data/processed_recipes.csv'
MODELS_DIR = 'models'

# --- KONFIGURASI TRAINING ---
config = {
    'max_features': 10000,
    'max_sequence_length': 512,
    'embedding_dim': 100,
    'lstm_units': 256,
    'learning_rate': 0.00005,
    'batch_size': 64,
    'dropout': 0.4,
    'recurrent_dropout': 0.4
}
EPOCHS = 30
PATIENCE = 5

if __name__ == "__main__":
    recommender = FoodRecommendationSystem(**config)

    # 1. MUAT DATA BERSIH DARI CSV
    print("Memuat data yang sudah diproses dari file CSV...")
    all_data = pd.read_csv(PROCESSED_DATA_PATH, delimiter=';')
    
    if 'loves' in all_data.columns:
        all_data['loves'] = all_data['loves'].fillna(0)

    # ================================================================= #
    # --- MULAI LOGIKA UNDERSAMPLING UNTUK MENYEIMBANGKAN DATASET --- #
    # ================================================================= #
    print("\nMenyeimbangkan dataset dengan metode undersampling...")
    
    # Tentukan kelas mayoritas dan minoritas
    category_counts = all_data['category'].value_counts()
    majority_class_name = category_counts.index[0]
    # Kita akan samakan jumlahnya dengan kelas kedua terbanyak, atau angka spesifik
    n_samples = category_counts.iloc[1] # Jumlah sampel kelas kedua terbanyak
    
    print(f"Kelas mayoritas adalah '{majority_class_name}' dengan {category_counts.iloc[0]} sampel.")
    print(f"Jumlah sampel akan disamakan menjadi ~{n_samples} untuk setiap kategori.")

    balanced_df_list = []
    for category in category_counts.index:
        df_category = all_data[all_data['category'] == category]
        # Jika kategori adalah mayoritas, kurangi sampelnya. Jika tidak, gunakan semua.
        df_resampled = resample(df_category, 
                                replace=False,    # Sampling tanpa penggantian
                                n_samples=min(len(df_category), n_samples), # Ambil n_samples atau semua data jika lebih sedikit
                                random_state=42) # random_state untuk hasil yang konsisten
        balanced_df_list.append(df_resampled)

    # Gabungkan kembali menjadi satu dataframe yang seimbang
    balanced_data = pd.concat(balanced_df_list)
    
    print("\nDistribusi data setelah diseimbangkan:")
    print(balanced_data['category'].value_counts())
    # ================================================================= #
    # --- SELESAI LOGIKA UNDERSAMPLING --- #
    # ================================================================= #


    # 2. LANJUTKAN KE FEATURE ENGINEERING DAN TRAINING MENGGUNAKAN DATA YANG SUDAH SEIMBANG
    print("\nMemulai feature engineering...")
    # Gunakan 'balanced_data' sebagai input, bukan 'all_data' lagi
    X, y = recommender.feature_engineering(balanced_data)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    recommender.build_model(len(recommender.label_encoder.classes_))
    history = recommender.train_model(X_train, y_train, epochs=EPOCHS, patience=PATIENCE)
    metrics, y_pred_classes = recommender.evaluate_model(X_test, y_test)
    
    # 3. SIMPAN MODEL & ARTIFAK LAINNYA
    training_timestamp = int(time.time())
    model_base_path = os.path.join(MODELS_DIR, f'model_{training_timestamp}')
    recommender.save_model(model_base_path)

    # Membuat file log secara manual
    TRAINING_LOGS_DIR = 'training_logs'
    os.makedirs(TRAINING_LOGS_DIR, exist_ok=True)
    cm = confusion_matrix(y_test, y_pred_classes)

    log_entry = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(training_timestamp)),
        "model_files": {"base_path": model_base_path},
        "parameters": {**config, "epochs": EPOCHS, "patience": PATIENCE},
        "metrics": metrics,
        "training_history": {k: [float(i) for i in v] for k, v in history.history.items()},
        "confusion_matrix": cm.tolist(),
        "parameters": {**config, "epochs": EPOCHS, "patience": PATIENCE, "class_labels": recommender.label_encoder.classes_.tolist()},
    }
    
    log_filename = os.path.join(TRAINING_LOGS_DIR, f"training_log_{training_timestamp}.json")
    with open(log_filename, 'w') as f:
        json.dump(log_entry, f, indent=4)
    print(f"✅ Log pelatihan berhasil disimpan di {log_filename}")

    print("\n✅ Proses pelatihan selesai.")