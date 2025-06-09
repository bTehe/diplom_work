# inference_page.py

import os
import base64
from datetime import datetime

import streamlit as st
import pandas as pd
import traceback

from utils.inference import infer_on_file, list_model_files
from utils.config import PROCESSED_FOLDER, MODEL_FOLDER, INFERENCE_FOLDER
import tensorflow as tf

def inference_page():
    st.title("🔍 Інференс")

    # ── Вхідні дані ───────────────────────────────────────────────────────────
    csv_files = [f for f in os.listdir(PROCESSED_FOLDER) if f.endswith(".csv")]
    model_files = list_model_files(MODEL_FOLDER)

    csv_choice = st.selectbox("Оброблений CSV-файл", [""] + csv_files)
    model_choice = st.selectbox("Файл моделі (.h5)", [""] + model_files)
    use_gpu = st.checkbox("Використовувати прискорення на GPU", value=True)

    if st.button("Запустити інференс"):
        if not csv_choice or not model_choice:
            st.error("Будь ласка, виберіть і CSV-файл, і файл моделі.")
            return

        with st.spinner("Виконується інференс…"):
            try:
                result = infer_on_file(
                    test_file=csv_choice,
                    processed_dir=PROCESSED_FOLDER,
                    model_path=os.path.join(MODEL_FOLDER, model_choice),
                    use_gpu=use_gpu
                )
                
                # Save results to CSV
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                results_df = pd.DataFrame({
                    'true': result['true_labels'],
                    'pred': result['predictions'],
                    'confidence': [max(p) for p in result.get('prediction_probs', [])]
                })

                metrics = result.get('metrics', {})
                for name, value in metrics.items():
                    results_df[name] = value

                results_df['model_used']   = result.get('model_used', model_choice)
                results_df['hardware']     = result.get('hardware', '—')
                results_df['inference_ts'] = timestamp

                output_file = f"{timestamp}_{os.path.splitext(csv_choice)[0]}_results.csv"
                output_path = os.path.join(INFERENCE_FOLDER, output_file)
                results_df.to_csv(output_path, index=False)
                st.success(f"Результати збережено у файл: {output_file}")
                
            except Exception as e:
                st.exception(e)
                tb = traceback.format_exc()
                st.error(f"Не вдалося виконати інференс:\n```\n{tb}\n```")
                st.error(f"Не вдалося виконати інференс: {e}")
                return

        _show_inference_results(result)


def _show_inference_results(r: dict):
    st.success("✅ Інференс завершено")
    st.subheader("Деталі виконання та зведення")
    st.write(f"**Використана модель:** {r.get('model_used', '—')}")
    st.write(f"**Апаратне забезпечення:** {r.get('hardware', '—')}")

    # Збираємо реальні метрики за вибіркою
    trues = r.get("true_labels", [])
    samples = len(trues)
    classes = len(set(trues))
    df_summary = pd.DataFrame({
        "Набір даних": ["Тестовий"],
        "Зразків": [samples],
        "Класів": [classes]
    })
    st.table(df_summary)

    st.subheader("Метрики")
    m = r.get("metrics", {})
    df_metrics = pd.DataFrame({
        "Метрика": [
            "Точність", "Precision (macro)", "Recall (macro)", "F1-оцінка (macro)",
            "Вагома F1", "Top-3 точність", "Top-5 точність",
            "Log-loss", "ROC AUC (micro)", "ROC AUC (macro)",
            "PR AUC (micro)", "PR AUC (macro)"
        ],
        "Значення": [
            m.get("accuracy"),
            m.get("precision"),
            m.get("recall"),
            m.get("f1"),
            m.get("weighted_f1"),
            m.get("top_3_accuracy") or "N/A",
            m.get("top_5_accuracy") or "N/A",
            m.get("log_loss"),
            m.get("roc_auc_micro") or "N/A",
            m.get("roc_auc_macro") or "N/A",
            m.get("pr_auc_micro") or "N/A",
            m.get("pr_auc_macro") or "N/A",
        ]
    })
    st.table(df_metrics)

    st.subheader("Візуалізації")
    c1, c2 = st.columns(2)
    if r.get("confusion_matrix_plot"):
        c1.image(
            base64.b64decode(r["confusion_matrix_plot"]),
            caption="Матриця невідповідностей",
            use_container_width=True
        )
    if r.get("roc_plot"):
        c2.image(
            base64.b64decode(r["roc_plot"]),
            caption="ROC-криві",
            use_container_width=True
        )
    c3, c4 = st.columns(2)
    if r.get("pr_plot"):
        c3.image(
            base64.b64decode(r["pr_plot"]),
            caption="Криві точність–відзив",
            use_container_width=True
        )
    if r.get("calibration_plot"):
        c4.image(
            base64.b64decode(r["calibration_plot"]),
            caption="Крива калібрування",
            use_container_width=True
        )

    st.subheader("Найпомітніші помилки")
    preds = r.get("predictions", [])
    trues = r.get("true_labels", [])
    errors = [
        {"Індекс": i, "Правильна мітка": trues[i], "Передбачена мітка": preds[i]}
        for i in range(len(trues)) if preds[i] != trues[i]
    ]
    if errors:
        st.table(pd.DataFrame(errors).head(10))
    else:
        st.write("Помилок класифікації не знайдено!")
