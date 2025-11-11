# streamlit_ui.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import pickle
import traceback
import seaborn as sns
from tensorflow.keras.models import load_model

# Coba impor kelas utama
try:
    from food_recommendation_system import FoodRecommendationSystem
except ImportError:
    st.error(
        "**Error:** Gagal mengimpor `food_recommendation_system.py`. Pastikan file tersebut berada di direktori yang sama dan semua library (seperti TensorFlow) sudah terinstal dengan benar."
    )
    st.stop()

# --- Konfigurasi Halaman & Path ---
st.set_page_config(page_title="Rekomendasi Makanan", page_icon="🍳", layout="wide")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAINING_LOGS_DIR = os.path.join(BASE_DIR, 'training_logs')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
DATA_DIR = os.path.join(BASE_DIR, 'data')

# --- Inisialisasi Session State ---
if 'recommender' not in st.session_state:
    st.session_state.recommender = FoodRecommendationSystem()
if 'active_model_label' not in st.session_state:
    st.session_state.active_model_label = None
if 'is_model_loaded' not in st.session_state:
    st.session_state.is_model_loaded = False
if 'recommendations' not in st.session_state:
    st.session_state.recommendations = None

# --- Fungsi Helper ---
@st.cache_data
def load_available_models():
    """
    MODIFIKASI: Memuat dan mengkategorikan model berdasarkan arsitekturnya.
    Return: Dictionary yang dikelompokkan, cth: {'Single Layer': {label: path}, 'Double Layer': {label: path}}
    """
    categorized_models = {}
    if not os.path.exists(TRAINING_LOGS_DIR):
        return categorized_models
    
    log_files = sorted([f for f in os.listdir(TRAINING_LOGS_DIR) if f.endswith('.json')], reverse=True)
    
    for log_file in log_files:
        try:
            with open(os.path.join(TRAINING_LOGS_DIR, log_file), 'r') as f:
                data = json.load(f)
                architecture = data.get('parameters', {}).get('model_architecture', 'Lainnya')
                accuracy = data.get('metrics', {}).get('accuracy', 0)
                timestamp = data.get('timestamp', 'N/A')
                
                # Buat label tanpa menyertakan arsitektur lagi
                label = f"Dilatih pada {timestamp.split(' ')[0]} | Akurasi: {accuracy:.2%}"
                
                if architecture not in categorized_models:
                    categorized_models[architecture] = {}
                
                if 'model_files' in data and 'base_path' in data['model_files']:
                    base_path = data['model_files']['base_path']
                    categorized_models[architecture][label] = base_path
        except Exception:
            continue
            
    return categorized_models

def display_recommendation(rec, index):
    """Fungsi untuk menampilkan satu item rekomendasi dengan format yang bagus."""
    with st.container(border=True):
        st.subheader(f"{index}. {rec['title']}")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            st.caption(f"Kategori Prediksi: **{rec['category']}**")
            if rec.get('url'):
                st.link_button("Lihat Resep Lengkap ↗️", rec['url'], use_container_width=True)
        with col2:
            st.metric(label="Tingkat Kemiripan", value=f"{rec['similarity']:.1%}")

        with st.expander("Lihat Bahan-Bahan"):
            ingredients_list = [ing.strip() for ing in rec['ingredients'].split('\n') if ing.strip()]
            for ingredient in ingredients_list:
                st.markdown(f"- {ingredient}")
def calculate_detailed_metrics_from_cm(cm_data, class_labels):
    """
    Menghitung metrik makro DAN metrik per kelas dari confusion matrix.
    """
    cm = np.array(cm_data)
    num_classes = len(cm)
    
    per_class_metrics = []
    
    # Hitung metrik untuk setiap kelas
    for i in range(num_classes):
        tp = cm[i, i]
        fp = np.sum(cm[:, i]) - tp
        fn = np.sum(cm[i, :]) - tp
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        per_class_metrics.append({
            "Kelas": class_labels[i],
            "Precision": precision,
            "Recall": recall,
            "F1-Score": f1
        })

    # Hitung rata-rata makro dari nilai yang sudah dihitung
    df = pd.DataFrame(per_class_metrics)
    macro_metrics = {
        "macro_precision": df["Precision"].mean(),
        "macro_recall": df["Recall"].mean(),
        "macro_f1_score": df["F1-Score"].mean()
    }
            
    return macro_metrics, df
# ======================================================================
# --- TAMPILAN UI ---
# ======================================================================

# --- Sidebar Navigasi ---
with st.sidebar:
    st.title("🍳 Foodie Finder")
    st.caption("Sistem Rekomendasi Makanan Cerdas")
    page = st.radio("Pilih Halaman:", ["🏠 Rekomendasi Makanan", "📊 Log Pelatihan"], label_visibility="collapsed")
    st.markdown("---")
    if st.session_state.is_model_loaded:
        st.success(f"Model Aktif:\n**{st.session_state.active_model_label}**")
    else:
        st.info("Belum ada model yang dimuat. Silakan pilih dan muat model di halaman utama.")

# --- HALAMAN REKOMENDASI ---
if page == "🏠 Rekomendasi Makanan":
    st.title("🔍 Cari Resep Makanan Favoritmu")
    st.markdown("Pilih tipe model AI, pilih versi yang ingin Anda gunakan, lalu masukkan preferensi makanan Anda.")

    categorized_models = load_available_models()

    if not categorized_models:
        st.error("Tidak ada model yang terlatih ditemukan. Harap jalankan training terlebih dahulu.")
    else:
        # --- MODIFIKASI: Bagian Kontrol Model dengan 2 Box ---
        with st.container(border=True):
            st.subheader("Pengaturan Model AI")
            col1, col2, col3 = st.columns([2, 3, 1])
            
            with col1:
                # Box 1: Pilih Tipe Model
                model_types = list(categorized_models.keys())
                selected_type = st.selectbox("1. Pilih Tipe Model", model_types)
            
            with col2:
                # Box 2: Pilih Model Spesifik berdasarkan Tipe
                models_in_type = categorized_models[selected_type]
                selected_label = st.selectbox("2. Pilih Versi Model", models_in_type.keys())
            
            with col3:
                # Tombol Muat diletakkan di paling kanan
                st.write("") # Trik untuk alignment vertikal
                st.write("")
                if st.button("Muat Model 🚀", use_container_width=True, type="primary"):
                    selected_base_path = models_in_type[selected_label]
                    full_label_for_display = f"{selected_type} ({selected_label.split('|')[0].strip()})"
    
                    # GANTI BLOK "with st.spinner(...)" DENGAN INI
                    with st.spinner("Memuat semua model pendukung..."):
                        try:
                             # Re-inisialisasi agar state bersih setiap kali memuat
                            recommender = FoodRecommendationSystem()
            
                             # 1. Muat model LSTM utama
                            model_basename = os.path.basename(selected_base_path)
                            recommender.load_model(os.path.join(MODELS_DIR, model_basename))
            
                            # 2. Muat model Word2Vec
                            recommender.load_w2v_model(os.path.join(MODELS_DIR, 'word2vec_recipes.model'))
            
                            # 3. Muat model TF-IDF
                            recommender.load_tfidf_model(
                                os.path.join(MODELS_DIR, 'tfidf_vectorizer.pkl'),
                                os.path.join(MODELS_DIR, 'tfidf_recipe_matrix.pkl')
                            )

                            # 4. Muat model Jaccard
                            recommender.load_jaccard_model(
                                os.path.join(MODELS_DIR, 'jaccard_vectorizer.pkl'),
                                os.path.join(MODELS_DIR, 'jaccard_recipe_matrix.pkl')
                            )
            
                            # 5. Muat data resep
                            recommender.load_processed_data(os.path.join(DATA_DIR, 'processed_recipes.csv'))
            
                            # Simpan semuanya ke session state
                            st.session_state.recommender = recommender
                            st.session_state.active_model_label = full_label_for_display
                            st.session_state.is_model_loaded = True
                            st.rerun()
                        except Exception as e:
                            st.error(f"Gagal memuat model: {e}")
                            traceback.print_exc()

        # --- Bagian Form & Hasil Rekomendasi (Tidak berubah) ---
        if st.session_state.is_model_loaded:
            st.markdown("---")
            st.subheader("✅ Model Siap! Silakan Cari Rekomendasi Anda")

            # PILIHAN METODE PERANGKINGAN MUNCUL DI SINI
            ranking_options = {
                "Cosine Similarity ": "cosine",
                "Jaccard Similarity ": "jaccard",
                "Euclidean Distance ": "euclidean",
                "Manhattan Distance ": "manhattan",
                "Dot Product ": "dot_product"
            }
            selected_method_label = st.selectbox(
                "Pilih Metode Perangkingan:",
                ranking_options.keys()
            )
            method_to_use = ranking_options[selected_method_label]

            with st.form("recommendation_form"):
                user_input = st.text_area(
                    "Masukkan preferensi makanan Anda (contoh: *resep telur balado sederhana*)",
                    height=100
                )
                submitted = st.form_submit_button("Cari Rekomendasi", use_container_width=True, type="primary")
                if submitted and user_input:
                    with st.spinner(f"Mencari dengan metode '{selected_method_label}'... 🤔"):
                        try:
                            recs = st.session_state.recommender.get_recommendations(
                                user_input, top_n=5, method=method_to_use
                        )
                            st.session_state.recommendations = recs
                        except Exception as e:
                            st.error(f"Terjadi kesalahan saat mencari rekomendasi: {e}")
    
    # Logika untuk menampilkan hasil (tidak berubah)
            if st.session_state.recommendations is not None:
                st.markdown("---")
                if not st.session_state.recommendations:
                    st.warning("Tidak ada rekomendasi yang cocok ditemukan. Coba gunakan kata kunci lain.")
                else:
                    for i, rec in enumerate(st.session_state.recommendations):
                        display_recommendation(rec, i + 1)
        else:
            st.info("👆 Silakan pilih dan muat model di atas untuk memulai.")


# --- HALAMAN LOG PELATIHAN (Tidak berubah) ---
elif page == "📊 Log Pelatihan":
    st.title("📊 Log dan Analisis Pelatihan Model")

    # Menggabungkan semua model untuk dropdown log
    all_models_flat = {}
    categorized_logs = load_available_models()
    for arch, models in categorized_logs.items():
        for label, path in models.items():
            full_label = f"{arch} | {label}"
            all_models_flat[full_label] = path

    if not all_models_flat:
        st.info("Belum ada log pelatihan yang tersimpan.")
    else:
        # Dropdown di sini tetap menampilkan semua agar mudah diakses
        selected_full_label = st.selectbox("Pilih Log untuk Dilihat Detailnya:", all_models_flat.keys())
        selected_base_path = all_models_flat[selected_full_label]
        
        log_timestamp = os.path.basename(selected_base_path).split('_')[1]
        log_filename = f"training_log_{log_timestamp}.json"
        log_filepath = os.path.join(TRAINING_LOGS_DIR, log_filename)

        if os.path.exists(log_filepath):
            with open(log_filepath, 'r') as f:
                log_data = json.load(f)

            st.header(f"Analisis Model: {selected_full_label.split('|')[0].strip()}")
            st.caption(f"Dilatih pada: {log_data.get('timestamp', 'N/A')}")
            
            tab1, tab2, tab3 = st.tabs(["📈 Performa & Metrik", "🧮 Confusion Matrix", "⚙️ Konfigurasi Model"])

            with tab1:
                st.subheader("Metrik Evaluasi Kunci (Macro Average)")
                
                # Ambil data yang dibutuhkan dari log
                original_metrics = log_data.get('metrics', {})
                cm_data = log_data.get('confusion_matrix')
                class_labels = log_data.get('parameters', {}).get('class_labels', [f"Kelas {i}" for i in range(len(cm_data))])

                if cm_data and class_labels:
                    # Hitung metrik makro dan per-kelas secara on-the-fly
                    macro_metrics, per_class_df = calculate_detailed_metrics_from_cm(cm_data, class_labels)
                    
                    # Tampilkan metrik utama
                    cols = st.columns(4)
                    cols[0].metric("Accuracy", f"{original_metrics.get('accuracy', 0):.2%}")
                    cols[1].metric("Macro Precision", f"{macro_metrics.get('macro_precision', 0):.2%}")
                    cols[2].metric("Macro Recall", f"{macro_metrics.get('macro_recall', 0):.2%}")
                    cols[3].metric("Macro F1-Score", f"{macro_metrics.get('macro_f1_score', 0):.2%}")

                    st.markdown("---") # Garis pemisah
                    
                    # # Tampilkan tabel rincian per kelas
                    # st.subheader("Rincian Metrik per Kelas")
                    # st.dataframe(per_class_df.style.format({
                    #     "Precision": "{:.2%}",
                    #     "Recall": "{:.2%}",
                    #     "F1-Score": "{:.2%}"
                    # }), use_container_width=True)

                else:
                    st.warning("Data confusion matrix atau label kelas tidak ditemukan di file log ini.")
                
                st.subheader("Grafik Performa Training")
                history = log_data.get('training_history', {})
                if history:
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
                    ax1.plot(history.get('accuracy', []), 'o-', label='Train Accuracy')
                    ax1.plot(history.get('val_accuracy', []), 'o-', label='Validation Accuracy')
                    ax1.set_title('Model Accuracy'); ax1.set_xlabel('Epoch'); ax1.legend(); ax1.grid(True)
                    
                    ax2.plot(history.get('loss', []), 'o-', label='Train Loss')
                    ax2.plot(history.get('val_loss', []), 'o-', label='Validation Loss')
                    ax2.set_title('Model Loss'); ax2.set_xlabel('Epoch'); ax2.legend(); ax2.grid(True)
                    st.pyplot(fig)

            with tab2:
                st.subheader("Analisis Confusion Matrix")
                st.caption("Sumbu Y adalah label asli, sumbu X adalah label prediksi model.")
                try:
                    cm_data = np.array(log_data['confusion_matrix'])
                    label_names = log_data.get('parameters', {}).get('class_labels', [])
                    
                    fig_cm, ax_cm = plt.subplots(figsize=(10, 8))
                    sns.heatmap(cm_data, annot=True, fmt='d', cmap='Blues', ax=ax_cm, 
                                xticklabels=label_names, yticklabels=label_names)
                    ax_cm.set_xlabel('Predicted Label')
                    ax_cm.set_ylabel('True Label')
                    ax_cm.set_title('Confusion Matrix')
                    st.pyplot(fig_cm)
                except Exception as e:
                    st.error(f"Gagal membuat confusion matrix: {e}")

            with tab3:
                st.subheader("Parameter & Konfigurasi")
                st.json(log_data.get('parameters', {}), expanded=False)
                with st.expander("Tampilkan Raw JSON Log Lengkap"):
                    st.json(log_data)