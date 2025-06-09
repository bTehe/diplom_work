# reports_page.py

import streamlit as st
import os

from utils.config import INFERENCE_FOLDER, REPORT_FOLDER
from utils.reports import (
    list_inference_results,
    list_reports,
    save_csv_report,
    save_pdf_report
)


def reports_page():
    st.title("📑 Звіти")

    inc_dir = INFERENCE_FOLDER
    report_dir = REPORT_FOLDER

    # Отримуємо список CSV-файлів з результатами inference
    results = list_inference_results(inc_dir)
    if not results:
        st.info("Файли результатів inference не знайдено.")
        return

    # Отримуємо вже створені звіти
    existing = list_reports(report_dir)

    # Форма для генерації нового звіту
    with st.form("generate_report"):
        result_file = st.selectbox(
            "Виберіть файл результатів inference:",
            options=[""] + results
        )
        report_type = st.selectbox(
            "Тип звіту:",
            options=["CSV", "PDF"]
        )
        submit = st.form_submit_button("Створити звіт")

    if submit:
        if not result_file:
            st.warning("Будь ласка, виберіть файл результатів.")
        else:
            try:
                if report_type == "CSV":
                    filename = save_csv_report(inc_dir, report_dir, result_file)
                else:
                    filename = save_pdf_report(inc_dir, report_dir, result_file)

                st.success(f'Звіт "{filename}" успішно створено.')
                # Оновлюємо список наявних звітів
                existing = list_reports(report_dir)
            except Exception as e:
                st.error(f"Помилка при створенні звіту: {e}")

    # Відображаємо доступні звіти для завантаження
    st.subheader("Доступні звіти")

    st.markdown("**CSV-звіти**")
    if existing['csv']:
        for fname in existing['csv']:
            path = os.path.join(report_dir, fname)
            with open(path, "rb") as f:
                data = f.read()
            st.download_button(
                label=f"Завантажити {fname}",
                data=data,
                file_name=fname,
                mime="text/csv"
            )
    else:
        st.write("CSV-звітів немає.")

    st.markdown("**PDF-звіти**")
    if existing['pdf']:
        for fname in existing['pdf']:
            path = os.path.join(report_dir, fname)
            with open(path, "rb") as f:
                data = f.read()
            st.download_button(
                label=f"Завантажити {fname}",
                data=data,
                file_name=fname,
                mime="application/pdf"
            )
    else:
        st.write("PDF-звітів немає.")
