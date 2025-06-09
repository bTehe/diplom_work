import streamlit as st
import os
from pathlib import Path

from utils.config import UPLOAD_FOLDER
from utils.data_loader import load_csv_preview

def upload_page():
    st.header("📥 Завантаження даних")

    # Переконуємося, що папка для завантажень існує
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)

    # Завантажувач файлів
    uploaded_file = st.file_uploader(
        "Оберіть CSV або PCAP файл:",
        type=["csv", "pcap"]
    )

    if uploaded_file is None:
        st.info("Завантажте файл, щоб розпочати.")
        return

    # Зберігаємо файл у вказаній папці
    save_path = Path(UPLOAD_FOLDER) / uploaded_file.name
    try:
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f'Файл "{uploaded_file.name}" збережено у `{UPLOAD_FOLDER}`.')
    except Exception as e:
        st.error(f"Не вдалося зберегти файл: {e}")
        return

    # Якщо це CSV — показуємо прев'ю
    if uploaded_file.name.lower().endswith(".csv"):
        try:
            preview_html = load_csv_preview(str(save_path), n=5)
            st.markdown("**Preview перших 5 рядків:**", unsafe_allow_html=True)
            st.markdown(preview_html, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Помилка при формуванні прев'ю: {e}")
    else:
        st.info("Прев'ю PCAP-файлів наразі не підтримується.")
