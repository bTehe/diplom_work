# app/utils/inference.py
import os
import numpy as np
import pandas as pd
from glob import glob
from flask import current_app
from tensorflow.keras.models import load_model

def list_model_files(model_dir: str) -> list[str]:
    return [fname for fname in os.listdir(model_dir) if fname.lower().endswith('.h5')]

def list_result_files(inc_dir: str) -> list[str]:
    pattern = os.path.join(inc_dir, "*_results.csv")
    return sorted(glob(pattern))

def predict_from_csv(
    filename: str,
    processed_dir: str,
    model_dir: str,
    top_n: int = 5
) -> dict:
    # Шляхи
    csv_path = os.path.join(processed_dir, filename)
    model_files = list_model_files(model_dir)
    if not model_files:
        raise FileNotFoundError("У MODEL_FOLDER немає .h5-моделей.")
    model_name = model_files[0]
    model_path = os.path.join(model_dir, model_name)
    model = load_model(model_path)

    # 1) Завантаження даних
    df = pd.read_csv(csv_path)
    X = df.drop(columns=['Label']).values

    # 2) Передбачення
    probs = model.predict(X, verbose=0)
    preds = np.argmax(probs, axis=1)

    # 3) Збір результатів
    classes = list(range(probs.shape[1]))
    results = pd.DataFrame(probs, columns=[f"p_{c}" for c in classes])
    results['pred'] = preds
    results['true'] = df['Label'].values
    # confidence = max probability
    results['confidence'] = results[[f"p_{c}" for c in classes]].max(axis=1)

    # 4) Збереження повних результатів
    out_fname = f"{os.path.splitext(filename)[0]}_{os.path.splitext(model_name)[0]}_results.csv"
    out_path  = os.path.join(current_app.config['INFERENCE_FOLDER'], out_fname)
    os.makedirs(current_app.config['INFERENCE_FOLDER'], exist_ok=True)
    results.to_csv(out_path, index=False)

    # 5) Підготовка preview топ-N помилок
    errors = results[results['pred'] != results['true']]
    top_errors = errors.sort_values('confidence', ascending=False).head(top_n)

    return {
        'model_used': model_name,
        'preview_input': df.head().to_html(classes="table table-sm", index=False),
        'errors_top':      top_errors.to_html(classes="table table-sm table-danger", index=False),
        'results_file':    out_fname
    }
