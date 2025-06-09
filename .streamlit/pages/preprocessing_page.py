# preprocessing_page.py

import os
import streamlit as st
import pandas as pd
from pathlib import Path

from utils.preprocessing import list_raw_files, preprocess_files_combined
from utils.config import UPLOAD_FOLDER, PROCESSED_FOLDER

def preprocessing_page():
    st.title("🧹 Попередня обробка та EDA")

    # 1) Список усіх “raw” CSV-файлів
    files = list_raw_files(UPLOAD_FOLDER)
    selected = st.multiselect(
        "Виберіть один або кілька RAW CSV-файлів",
        options=files,
        help="Використовуйте Ctrl/Cmd для множинного вибору"
    )

    # 2) Кнопка запуску обробки
    if st.button("Запустити попередню обробку"):
        if not selected:
            st.error("Будь ласка, виберіть принаймні один файл для обробки.")
            return

        with st.spinner("Очищення, перетворення та розбиття…"):
            try:
                summaries = preprocess_files_combined(
                    filenames=selected,
                    raw_dir=UPLOAD_FOLDER,
                    processed_dir=PROCESSED_FOLDER
                )
            except Exception as e:
                st.error(f"Помилка під час попередньої обробки:\n{e}")
                return

        st.success("✅ Попередня обробка завершена")

        # 3) Відображення результатів
        for s in summaries:
            st.subheader(f"Файл: {s.filename}")
            st.markdown(f"""
- **Початкові рядки:** {s.initial_rows}
- **Рядки після очищення:** {s.after_clean_rows}
- **Відсутні значення до очищення:** {s.missing_before}
- **Негативні значення до очищення:** {s.negatives_before}
- **Папка з обробленими файлами:** `{s.processed_path}`
            """)

        # 4) Перегляд перших 5 рядків обробленого DataFrame
        if summaries:
            st.subheader("Попередній перегляд оброблених даних (перші 5 рядків)")
            st.markdown(summaries[0].df_head, unsafe_allow_html=True)
