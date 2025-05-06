# app/utils/preprocessing.py
import os
import numpy as np
import pandas as pd
from glob import glob

def list_raw_files(raw_dir: str) -> list[str]:
    """Повертає список CSV-файлів у папці raw."""
    pattern = os.path.join(raw_dir, "*.csv")
    return [os.path.basename(p) for p in glob(pattern)]

def preprocess_file(
    filename: str,
    raw_dir:      str,
    processed_dir:str
) -> dict:
    """
    Завантажує raw CSV, робить:
     - strip назв колонок
     - заміну ±inf → NaN, видалення рядків з NaN
     - видалення рядків з будь-якими від’ємними числами
     - (за потреби) інженерію часових ознак, якщо є 'timestamp'
     - збереження в processed_dir
    Повертає словник summary з кількістю рядків, шляхом до файлу, preview HTML і distribution HTML.
    """
    raw_path = os.path.join(raw_dir, filename)
    df = pd.read_csv(raw_path)
    summary = {}

    # 1) Загальна інформація
    summary['initial_rows'], summary['initial_cols'] = df.shape

    # 2) strip spaces
    df.columns = df.columns.str.strip()

    # 3) ±inf → NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    summary['missing_before'] = int(df.isna().sum().sum())

    # 4) видалити NaN
    df.dropna(inplace=True)

    # 5) знайти негативні у числових
    num_cols = df.select_dtypes(include=[np.number]).columns
    neg_mask = (df[num_cols] < 0).any(axis=1)
    summary['negatives_before'] = int(neg_mask.sum())

    # 6) видалити негативні
    df = df.loc[~neg_mask]
    summary['after_clean_rows'] = df.shape[0]

    # 7) інженерія часових ознак
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['dow'] = df['timestamp'].dt.dayofweek
        df['hour']= df['timestamp'].dt.hour
        df['dow_sin'] = np.sin(2*np.pi*df['dow']/7)
        df['dow_cos'] = np.cos(2*np.pi*df['dow']/7)
        df['hour_sin']= np.sin(2*np.pi*df['hour']/24)
        df['hour_cos']= np.cos(2*np.pi*df['hour']/24)

    # 8) збереження
    os.makedirs(processed_dir, exist_ok=True)
    out_path = os.path.join(processed_dir, filename)
    df.to_csv(out_path, index=False)
    summary['processed_path'] = out_path

    # 9) preview HTML
    summary['preview'] = df.head().to_html(
        classes="table table-sm table-striped", index=False
    )

    # 10) distribution HTML (якщо є 'Label')
    if 'Label' in df.columns:
        dist = (
            df['Label']
            .value_counts()
            .rename_axis('Label')
            .reset_index(name='Count')
        )
        summary['distribution'] = dist.to_html(
            classes="table table-sm", index=False
        )

    return summary