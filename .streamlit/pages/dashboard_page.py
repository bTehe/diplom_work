# dashboard_page.py

import streamlit as st
from pathlib import Path

# Шляхи до папок
BASE_DIR = Path(".streamlit/utils")
DATA_DIR = BASE_DIR / "data"
UPLOAD_FOLDER = DATA_DIR / "raw"
PROCESSED_FOLDER = DATA_DIR / "processed"
MODEL_FOLDER = BASE_DIR / "models"
INFERENCE_FOLDER = DATA_DIR / "inference_results"
REPORT_FOLDER = BASE_DIR / "reports"

def count_files(directory: Path, pattern: str = "*") -> int:
    """Count files in directory matching pattern"""
    if not directory.exists():
        return 0
    return len(list(directory.glob(pattern)))

def dashboard_page():
    st.title("🛡️ Система виявлення кібер-атак")
    st.write("Оперативний моніторинг і аналітика на основі глибинного навчання.")

    # Гарантуємо, що папки є
    for folder in [UPLOAD_FOLDER, PROCESSED_FOLDER, MODEL_FOLDER, INFERENCE_FOLDER, REPORT_FOLDER]:
        folder.mkdir(parents=True, exist_ok=True)

    # Підрахунок статистики
    stats = {
        "raw_files": count_files(UPLOAD_FOLDER),
        "processed_files": count_files(PROCESSED_FOLDER),
        "models": count_files(MODEL_FOLDER, "*.h5"),
        "inference_runs": count_files(INFERENCE_FOLDER),
        "reports": count_files(REPORT_FOLDER),
    }

    # Виводимо всі 5 метрик в одній стрічці з 5 колонок
    cols = st.columns(5)
    cols[0].metric(label="📁 Завантажені файли", value=stats["raw_files"], help="очікують обробки")
    cols[1].metric(label="⚙️ Оброблені дані", value=stats["processed_files"], help="готові до тренування")
    cols[2].metric(label="🤖 Навчені моделі", value=stats["models"], help="доступні для inference")
    cols[3].metric(label="📊 Запуски inference", value=stats["inference_runs"], help="результати в історії")
    cols[4].metric(label="📝 Звіти", value=stats["reports"], help="CSV & PDF")

    st.markdown("---")
    st.info(
        "Порада: перейдіть у розділ «Preprocessing», щоб завантажити й обробити дані, "
        "або в «Training», щоб натренувати модель на оброблених даних."
    )
