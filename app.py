import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Fungsi untuk load model (gunakan @st.cache agar tidak bolak-balik load)
@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model("model_tomat.keras", compile=False)  
    return model

# Load model sekali saja
model = load_model()

# Judul aplikasi
st.title("🍅 Deteksi Penyakit Daun Tomat dengan CNN")
st.markdown("""
Aplikasi ini dibuat untuk **mendeteksi penyakit daun tomat** menggunakan model 
*Convolutional Neural Network (CNN)*.  
Saat ini, model dapat mengklasifikasikan gambar daun tomat ke dalam **2 kelas utama**:

1. **Bacterial Spot** → Penyakit yang disebabkan oleh bakteri *Xanthomonas perforans* dan *Xanthomonas vesicatoria*, 
   ditandai dengan bercak kecil berwarna gelap pada daun.
2. **Early Blight** → Penyakit yang disebabkan oleh jamur *Alternaria solani*, 
   ditandai dengan bercak coklat melingkar konsentris pada daun.""")

# Upload gambar
uploaded_file = st.file_uploader("Upload gambar daun tomat", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Tampilkan gambar
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Gambar yang diupload", use_column_width=True)

    # Preprocessing sesuai input model (misalnya 256x256)
    img = image.resize((256, 256))  
    img_array = np.array(img) / 255.0  
    img_array = np.expand_dims(img_array, axis=0)  

    # Prediksi
    prediction = model.predict(img_array)[0][0]

    # Tampilkan hasil
    if prediction < 0.5:
        kelas = "Bacterial Spot"
        confidence = 100.0
    else:
        kelas = "Early Blight"
        confidence = 100.0

    st.success(f"Hasil Prediksi: **{kelas}** ({confidence:.2f}%)")