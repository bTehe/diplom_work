# app/utils/monitoring.py
import os
import pandas as pd
from glob import glob

def list_incident_files(inc_dir: str) -> list[str]:
    """Перелік CSV із результатами inference."""
    pattern = os.path.join(inc_dir, "*_results.csv")
    return [os.path.basename(p) for p in sorted(glob(pattern))]

def load_incidents(file_name: str, inc_dir: str) -> pd.DataFrame:
    """Завантажує один файл результатів."""
    path = os.path.join(inc_dir, file_name)
    return pd.read_csv(path)

def filter_incidents(
    df: pd.DataFrame,
    pred_class: int | None = None,
    min_confidence: float = 0.0
) -> pd.DataFrame:
    """Фільтрує за передбаченим класом і мінімальною довірою."""
    if pred_class is not None:
        df = df[df['pred'] == pred_class]
    if min_confidence > 0:
        df = df[df['confidence'] >= min_confidence]
    return df
