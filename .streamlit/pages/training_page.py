# сторінка_тренування.py

import os
import time
import json
import threading
import base64

import streamlit as st
import pandas as pd

from utils.training import train_model_on_files
from utils.config import PROCESSED_FOLDER, MODEL_FOLDER, TEMP_FOLDER

def training_page():
    st.title("🏋️ Тренування моделі")

    # ── Вибір файлів ─────────────────────────────────────
    csvs = [f for f in os.listdir(PROCESSED_FOLDER) if f.endswith(".csv")]
    train_csv = st.selectbox("Навчальний CSV", [""] + csvs)
    val_csv   = st.selectbox("Валідаційний CSV", [""] + csvs)
    test_csv  = st.selectbox("Тестовий CSV", [""] + csvs)

    # ── Розклад і апаратне забезпечення ───────────────────
    st.subheader("Розклад і апаратне забезпечення")
    batch_size = st.number_input("Розмір пакету", min_value=1, value=64)
    epochs     = st.number_input("Епохи",       min_value=1, value=20)
    lr         = st.number_input("Швидкість навчання", format="%.5f", value=0.001)
    use_gpu    = st.checkbox("Використовувати прискорення GPU", value=True)

    # ── Гіперпараметри моделі ────────────────────────────
    st.subheader("Гіперпараметри моделі")
    filters           = st.number_input("Кількість фільтрів конволюції",   value=64)
    kernel_size       = st.number_input("Розмір ядра",                      value=5)
    pool_size         = st.number_input("Розмір підвибірки",                value=2)
    lstm_units        = st.number_input("Кількість одиниць LSTM",           value=128)
    lstm_layers       = st.number_input("Шари LSTM",                       min_value=1, value=1)
    dropout_rate      = st.slider("Рівень відсіву",                       0.0, 1.0, 0.3)
    recurrent_dropout = st.slider("Рекурентний відсів",                  0.0, 1.0, 0.1)

    model_params = {
        "filters":           filters,
        "kernel_size":       kernel_size,
        "pool_size":         pool_size,
        "lstm_units":        lstm_units,
        "lstm_layers":       lstm_layers,
        "dropout_rate":      dropout_rate,
        "recurrent_dropout": recurrent_dropout,
        "learning_rate":     lr,
    }

    # ── Кнопка запуску ───────────────────────────────────
    if st.button("Розпочати навчання"):
        # перевірка вибору файлів
        if not (train_csv and val_csv and test_csv):
            st.error("Будь ласка, виберіть файли CSV для навчання, валідації та тестування.")
            return

        # очищаємо старий файл прогресу
        progress_file = os.path.join(TEMP_FOLDER, "training_progress.json")
        if os.path.exists(progress_file):
            os.remove(progress_file)

        # запускаємо навчання в окремому потоці
        args = dict(
            train_file=train_csv,
            val_file=val_csv,
            test_file=test_csv,
            processed_dir=PROCESSED_FOLDER,
            model_dir=MODEL_FOLDER,
            batch_size=batch_size,
            epochs=epochs,
            model_params=model_params,
            use_gpu=use_gpu,
            progress_file=progress_file,
        )
        threading.Thread(
            target=lambda: train_model_on_files(**args),
            daemon=True
        ).start()

        # ── Відображаємо прогрес ─────────────────────────
        bar    = st.progress(0)
        status = st.empty()

        while True:
            if os.path.exists(progress_file):
                try:
                    with open(progress_file) as f:
                        data = json.load(f)
                except json.JSONDecodeError:
                    time.sleep(0.5)
                    continue

                pct     = data.get("percentage", 0)
                metrics = data.get("metrics", {})

                bar.progress(pct)
                status.text(data.get("message", ""))

                # теперь проверяем, есть ли уже test_loss
                if pct >= 100 and metrics and ("test_loss" in metrics):
                    results = metrics
                    break

            else:
                status.text("Очікування початку навчання…")

            time.sleep(1)

        st.success("✅ Навчання завершено!")
        _show_results(results)


def _show_results(m):
    st.subheader("Підсумки навчання")
    st.write(f"**Апаратне забезпечення:** {m.get('hardware')}")
    st.write(f"**Розмір пакету:** {m.get('batch_size')}")
    st.markdown(m.get("summary", ""), unsafe_allow_html=True)

    st.subheader("Кінцеві метрики тестування")
    st.write(f"- Помилка на тесті: `{m.get('test_loss'):.4f}`")
    st.write(f"- Точність на тесті: `{m.get('test_accuracy'):.4f}`")

    # графіки навчання
    c1, c2 = st.columns(2)
    if m.get("loss_plot"):
        c1.image(
            base64.b64decode(m["loss_plot"]),
            caption="Помилка по епохах", use_container_width=True
        )
    if m.get("accuracy_plot"):
        c2.image(
            base64.b64decode(m["accuracy_plot"]),
            caption="Точність по епохах", use_container_width=True
        )

    # повна історія
    st.subheader("Історія навчання")
    hist = m.get("history", {})
    if hist:
        df_hist = pd.DataFrame({
            "train_loss": hist.get("loss", []),
            "val_loss":   hist.get("val_loss", []),
            "train_acc":  hist.get("accuracy", []),
            "val_acc":    hist.get("val_accuracy", []),
        })
        st.line_chart(df_hist)

    # кнопка завантаження моделі
    model_path = m.get("model_path")
    if model_path and os.path.exists(model_path):
        with open(model_path, "rb") as f:
            st.download_button(
                "📥 Завантажити натреновану модель",
                data=f,
                file_name=os.path.basename(model_path),
                mime="application/octet-stream"
            )
