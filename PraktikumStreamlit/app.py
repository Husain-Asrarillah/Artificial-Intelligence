import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import time

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Spam Detector AI", page_icon="🚫", layout="wide")

# --- FUNGSI LOAD & TRAIN MODEL ---
@st.cache_resource
def train_model(file_path):
    # Membaca dataset
    df = pd.read_csv(file_path)
    
    # Menghapus baris kosong jika ada
    df.dropna(inplace=True)
    
    # Menentukan fitur dan target (Sesuaikan nama kolom dengan file CSV-mu)
    # Berdasarkan filemu: Class (label) dan Message (text)
    X = df['Message']
    y = df['Class']
    
    # Split data (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Membuat Pipeline: Vektorisasi Teks + Algoritma Naive Bayes
    # Naive Bayes sangat handal untuk klasifikasi teks/spam
    model = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('nb', MultinomialNB())
    ])
    
    # Training
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    
    return model, accuracy, df

# Memanggil fungsi training
# Pastikan nama file sesuai dengan yang kamu upload ke folder yang sama
try:
    model, accuracy, df = train_model("Spam_SMS.csv")
except Exception as e:
    st.error(f"Gagal memuat file CSV. Pastikan file 'Spam_SMS.csv' ada di folder yang sama. Error: {e}")
    st.stop()

# --- SIDEBAR ---
st.sidebar.title("🛡️ Detektor Spam SMS")
st.sidebar.markdown(f"""
**Informasi Model:**
- **Algoritma:** Multinomial Naive Bayes
- **Akurasi:** `{accuracy*100:.2f}%`
- **Total Data:** `{len(df)}` baris
""")

st.sidebar.divider()
st.sidebar.write("Contoh Pesan Spam:")
st.sidebar.caption("'CONGRATULATIONS! You won a $1000 Walmart gift card. Click here to claim...'")

# --- MAIN PAGE ---
st.title("✉️ AI Message Classifier")
st.write("Gunakan aplikasi ini untuk mengecek apakah sebuah pesan terindikasi sebagai spam atau pesan normal.")

# Input Pesan dari User
user_input = st.text_area("Masukkan Pesan SMS di sini:", placeholder="Tulis atau tempel pesan yang ingin diperiksa...", height=150)

# Tombol Analisis
col1, col2, col3 = st.columns([1,1,1])
with col2:
    predict_btn = st.button("🔍 Analisis Pesan", use_container_width=True, type="primary")

st.divider()

if predict_btn:
    if user_input.strip() == "":
        st.warning("Silakan masukkan teks terlebih dahulu!")
    else:
        with st.spinner('Menganalisis pola teks...'):
            time.sleep(1)
            prediction = model.predict([user_input])[0]
            probability = model.predict_proba([user_input])
            
        # Tampilan Hasil
        if prediction.lower() == 'spam':
            st.error(f"### 🚩 Hasil: **{prediction.upper()}**")
            st.markdown("⚠️ **Peringatan:** Pesan ini memiliki ciri-ciri penipuan atau spam.")
        else:
            st.success(f"### ✅ Hasil: **{prediction.upper()} (HAM)**")
            st.markdown("Pesan ini terlihat aman dan normal.")
            
        # Menampilkan Probabilitas (Keyakinan AI)
        st.write("**Tingkat Keyakinan AI:**")
        prob_df = pd.DataFrame(probability, columns=model.classes_)
        st.dataframe(prob_df.style.format("{:.2%}"), use_container_width=True)

# --- BAGIAN DATASET ---
with st.expander("📊 Lihat Data Referensi (Dataset)"):
    st.dataframe(df.head(100), use_container_width=True)
