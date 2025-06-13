import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# --- Konfigurasi Halaman ---
st.set_page_config(
    page_title="Klasifikasi Sampah Plastik",
    page_icon="♻️",
    layout="wide"
)

# --- Data Deskripsi Plastik ---
# Sama seperti versi web, kita gunakan dictionary ini untuk penjelasan
plastic_descriptions = {
    'PET': 'Biasa digunakan untuk botol air mineral, jus, dan minuman ringan. Dapat didaur ulang menjadi serat untuk pakaian atau karpet.',
    'HDPE': 'Digunakan untuk botol susu, botol deterjen, dan pipa. Dikenal kuat dan tahan terhadap bahan kimia. Dapat didaur ulang menjadi bangku taman atau pot tanaman.',
    'PVC': 'Sering digunakan untuk pipa, bingkai jendela, dan beberapa kemasan. Daur ulangnya lebih sulit.',
    'LDPE': 'Ditemukan pada kantong plastik, plastik pembungkus (wrap), dan botol yang bisa diremas. Dapat didaur ulang menjadi kantong sampah.',
    'PP': 'Digunakan untuk wadah makanan, tutup botol, dan komponen otomotif. Tahan panas. Dapat didaur ulang menjadi sikat atau baterai mobil.',
    'PS': 'Dikenal sebagai styrofoam, digunakan untuk cangkir kopi sekali pakai dan kemasan makanan. Sulit didaur ulang.',
    'Lainnya': 'Kategori untuk jenis plastik lain atau kombinasi beberapa jenis. Biasanya paling sulit untuk didaur ulang.'
}

# --- Memuat Model ---
# Streamlit memiliki cache untuk mencegah model dimuat ulang setiap kali ada interaksi.
@st.cache_resource
def load_model():
    """
    Fungsi untuk memuat model Keras (.h5) dari file.
    Ganti 'path_ke_model_anda.h5' dengan path file model Anda.
    Pastikan file model berada di folder yang sama dengan app.py atau sediakan path lengkapnya.
    """
    try:
        # GANTI NAMA FILE INI DENGAN NAMA FILE MODEL ANDA
        model = tf.keras.models.load_model('model_sampah.h5')
        return model
    except Exception as e:
        st.error(f"Error memuat model: {e}")
        st.error("Pastikan file 'model_sampah.h5' ada di folder yang sama dengan aplikasi ini.")
        return None

# Panggil fungsi untuk memuat model
model = load_model()

# --- Fungsi Prediksi ---
def process_image_and_predict(image_data, model_to_use):
    """
    Memproses gambar yang diunggah dan melakukan prediksi menggunakan model.
    """
    if model_to_use is None:
        return None, None

    # Buka gambar menggunakan Pillow
    image = Image.open(image_data).convert('RGB')
    
    # Ubah ukuran gambar sesuai input model (misal: 224x224)
    image = image.resize((224, 224))
    
    # Konversi gambar ke numpy array dan normalisasi
    image_array = np.array(image)
    image_array = image_array / 255.0
    
    # Tambah dimensi untuk mencocokkan input model (1, 224, 224, 3)
    image_array = np.expand_dims(image_array, axis=0)
    
    # Lakukan prediksi
    predictions = model_to_use.predict(image_array)
    
    # Dapatkan hasil
    score = np.max(predictions)
    # Ganti list ini dengan kelas yang sesuai dengan model Anda
    class_names = ['PET', 'HDPE', 'PVC', 'LDPE', 'PP', 'PS', 'Lainnya'] 
    predicted_class = class_names[np.argmax(predictions)]
    
    return predicted_class, score

# --- Inisialisasi State Aplikasi ---
if 'page' not in st.session_state:
    st.session_state.page = 'home'

# --- Tampilan Halaman (Navigasi) ---

# Halaman Awal (Home)
if st.session_state.page == 'home':
    st.title("Selamat Datang di Aplikasi Klasifikasi Sampah Plastik ♻️")
    st.markdown("---")
    st.write("Aplikasi ini membantu Anda mengidentifikasi berbagai jenis sampah plastik dengan mudah. Mari kita mulai menjaga lingkungan bersama.")
    
    if st.button("Mulai Pindai", type="primary", use_container_width=True):
        st.session_state.page = 'scan'
        st.rerun() # Muat ulang halaman untuk pindah ke page 'scan'

# Halaman Pindai (Scan)
elif st.session_state.page == 'scan':
    st.title("Pindai Sampah Anda")
    if st.button("⬅️ Kembali ke Halaman Awal", use_container_width=True):
        st.session_state.page = 'home'
        st.rerun()
    st.markdown("---")
    
    # Opsi Input
    tab1, tab2 = st.tabs(["Unggah Gambar", "Ambil Foto dengan Kamera"])

    with tab1:
        uploaded_file = st.file_uploader("Pilih sebuah gambar...", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            # Tampilkan gambar yang diunggah
            st.image(uploaded_file, caption="Gambar yang Diunggah", use_column_width=True)
            
            # Lakukan prediksi saat tombol ditekan
            if st.button("Klasifikasi Gambar Ini", use_container_width=True):
                if model:
                    with st.spinner('Sedang menganalisis gambar...'):
                        predicted_class, score = process_image_and_predict(uploaded_file, model)
                        # Simpan hasil ke session state untuk ditampilkan di halaman hasil
                        st.session_state.prediction_result = (predicted_class, score, uploaded_file)
                        st.session_state.page = 'result'
                        st.rerun()
                else:
                    st.warning("Model tidak dapat dimuat, fitur klasifikasi tidak tersedia.")

    with tab2:
        camera_input = st.camera_input("Arahkan kamera ke sampah plastik")
        if camera_input is not None:
             # Tampilkan gambar dari kamera
            st.image(camera_input, caption="Gambar dari Kamera", use_column_width=True)

            if st.button("Klasifikasi Foto Ini", use_container_width=True):
                if model:
                    with st.spinner('Sedang menganalisis foto...'):
                        predicted_class, score = process_image_and_predict(camera_input, model)
                        st.session_state.prediction_result = (predicted_class, score, camera_input)
                        st.session_state.page = 'result'
                        st.rerun()
                else:
                    st.warning("Model tidak dapat dimuat, fitur klasifikasi tidak tersedia.")

# Halaman Hasil (Result)
elif st.session_state.page == 'result':
    st.title("Hasil Klasifikasi")
    st.markdown("---")
    
    # Ambil hasil dari session state
    predicted_class, score, image = st.session_state.get('prediction_result', (None, None, None))
    
    if predicted_class is None:
        st.warning("Tidak ada hasil untuk ditampilkan.")
    # Jika keyakinan model rendah, tampilkan peringatan
    elif score < 0.60: # Threshold bisa disesuaikan
        st.error("Gambar Tidak Valid atau Tidak Dikenali")
        st.image(image, caption="Gambar yang Dianalisis", width=400)
        st.warning(f"Model kami tidak yakin dengan gambar ini (keyakinan hanya {score:.2%}). Sepertinya ini bukan sampah plastik yang dapat dikenali. Silakan coba gambar lain.")
    else:
        st.success(f"**Jenis Plastik:** {predicted_class}")
        st.image(image, caption="Gambar yang Dianalisis", width=400)
        
        st.metric(label="Tingkat Keyakinan Model", value=f"{score:.2%}")
        
        # Tampilkan deskripsi
        st.subheader("Deskripsi dan Penanganan:")
        description = plastic_descriptions.get(predicted_class, "Deskripsi tidak tersedia.")
        st.info(description)

    if st.button("Pindai Lagi", type="primary", use_container_width=True):
        st.session_state.page = 'scan'
        st.rerun()
