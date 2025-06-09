# monitoring_page.py
import streamlit as st
import os
import pandas as pd
from utils.monitoring import (
    list_incident_files, load_incidents, filter_incidents,
    get_incident_statistics,
    get_anomaly_incidents, export_incidents
)
from utils.config import INFERENCE_FOLDER

def monitoring_page():
    """
    Сторінка Streamlit для моніторингу інцидентів інференсу з можливістю фільтрації.
    """
    st.title("Моніторинг та сповіщення")

    # Каталог з CSV-файлами результатів інференсу
    inc_dir = INFERENCE_FOLDER

    # Отримати список файлів інцидентів
    files = list_incident_files(inc_dir)
    if not files:
        st.info("У каталозі інференсу не знайдено файлів інцидентів.")
        return

    # Вибір файлу інцидентів
    selected_file = st.selectbox(
        "Оберіть файл інцидентів:",
        options=[""] + files,
        index=0
    )

    if selected_file:
        try:
            df = load_incidents(selected_file, inc_dir)
        except Exception as e:
            st.error(f"Помилка завантаження інцидентів: {e}")
            return

        # Створити вкладки для різних переглядів
        tab1, tab2, tab3 = st.tabs(["Відфільтрований перегляд", "Статистика", "Аномалії"])

        with tab1:
            # Параметри фільтрації
            pred_choices = ["Усі"] + sorted(df['pred'].unique().tolist())
            pred_choice = st.selectbox("Передбачений клас (за бажанням):", options=pred_choices)
            min_conf = st.slider(
                "Мінімальна впевненість:",
                min_value=0.0,
                max_value=1.0,
                value=0.0,
                step=0.01
            )

            # Застосувати фільтри та показати таблицю
            if st.button("Фільтрувати"):
                pred_class = None if pred_choice == "Усі" else int(pred_choice)
                df_filt = filter_incidents(
                    df,
                    pred_class=pred_class,
                    min_confidence=min_conf
                )
                if df_filt.empty:
                    st.info("Немає інцидентів, що відповідають заданим фільтрам.")
                else:
                    st.dataframe(df_filt, use_container_width=True)

                    # Опції експорту
                    export_format = st.selectbox("Формат експорту:", ["csv", "excel"])
                    if st.button("Експортувати дані"):
                        filename = export_incidents(df_filt, export_format)
                        st.success(f"Дані експортовано у файл {filename}")

        with tab2:
            # Показати статистику
            stats = get_incident_statistics(df)
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Загальна кількість інцидентів", stats['total_incidents'])
                st.metric("Середня впевненість", f"{stats['avg_confidence']:.2%}")
            with col2:
                st.metric("Мінімальна впевненість", f"{stats['min_confidence']:.2%}")
                st.metric("Максимальна впевненість", f"{stats['max_confidence']:.2%}")

            # Розподіл за класами
            st.subheader("Розподіл за класами")
            class_df = pd.DataFrame.from_dict(stats['class_distribution'], orient='index', columns=['кількість'])
            st.bar_chart(class_df)

        with tab3:
            # Виявлення аномалій
            st.subheader("Потенційні аномалії")
            anomaly_threshold = st.slider(
                "Поріг впевненості для аномалій:",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.01
            )
            anomalies = get_anomaly_incidents(df, anomaly_threshold)
            if not anomalies.empty:
                st.dataframe(anomalies, use_container_width=True)
            else:
                st.info("За поточним порогом аномалій не знайдено.")
