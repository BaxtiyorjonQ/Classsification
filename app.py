import streamlit as st
from fastai.vision.all import *
import pathlib
import plotly.express as px

# Pathlib muvofiqligi uchun sozlash
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

st.title('Transportni klassifikatsiya qiluvchi model')

# Fayl yuklash
file = st.file_uploader('Rasm yuklang', type=['png', 'jpeg', 'jpg', 'jfif'])

if file is not None:
    try:
        st.image(file)
        # Rasmni yuklash
        img = PILImage.create(file)

        # Modelni yuklash
        model = load_learner('transport_model.pkl')

        # Bashorat qilish
        pred, pred_id, probs = model.predict(img)
        
        # Natijalarni ko‘rsatish
        st.success(f"Bashorat: {pred}")
        st.info(f"Ehtimollik: {probs[pred_id]*100:.1f}%")
        
        # Ehtimolliklarni grafikda ko‘rsatish
        fig = px.bar(
            x=model.dls.vocab,  # Sinflar nomi
            y=probs * 100,      # Ehtimollik foizlari
            labels={'x': 'Sinflar', 'y': 'Ehtimollik (%)'},  # O‘qlar uchun yorliqlar
            title='Sinflar bo‘yicha ehtimollik'
        )
        st.plotly_chart(fig)

    except Exception as e:
        st.error(f"Xatolik yuz berdi: {e}")
else:
    st.warning("Iltimos, fayl yuklang.")
