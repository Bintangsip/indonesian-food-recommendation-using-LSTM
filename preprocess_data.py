import os
import pandas as pd
from food_recommendation_system import FoodRecommendationSystem

# --- PATHS ---
# Direktori tempat file-file CSV mentah berada
DATA_DIR = 'data' 
# Lokasi dan nama file output yang sudah bersih (diubah ke .csv)
OUTPUT_FILE = os.path.join(DATA_DIR, 'processed_recipes.csv') 

if __name__ == "__main__":
    # Pastikan direktori data ada
    if not os.path.exists(DATA_DIR):
        print(f"Error: Direktori '{DATA_DIR}' tidak ditemukan.")
    else:
        # Inisialisasi sistem
        recommender = FoodRecommendationSystem()
        
        # Kumpulkan informasi dataset mentah, KECUALIKAN file hasil proses sebelumnya
        dataset_info = [
            (os.path.join(DATA_DIR, f), 'utf-8') 
            for f in os.listdir(DATA_DIR) 
            if f.endswith('.csv') and 'processed_recipes.csv' not in f
        ]
        
        if not dataset_info:
            print("Tidak ada file .csv mentah yang ditemukan di dalam direktori 'data'.")
        else:
            # 1. Jalankan proses pembersihan data
            print("Memulai proses pembersihan dan stemming data...")
            processed_df = recommender.load_all_datasets(dataset_info)
            
            # 2. Simpan hasilnya ke format CSV
            if not processed_df.empty:
                print(f"Menyimpan data yang sudah diproses ke {OUTPUT_FILE}...")
                # Gunakan to_csv dengan separator ';'
                processed_df.to_csv(OUTPUT_FILE, index=False, sep=';', encoding='utf-8')
                print(f"✅ Data bersih berhasil disimpan.")