"""
Everything you need to clean a single raw CIC-IDS-2017 CSV and save the result
to `PROCESSED_DIR`.

The public surface is just two helpers:

    list_raw_files(raw_dir)                →  list[str]
    preprocess_files(filename, raw_dir,
                     processed_dir)        →  PreprocessSummary

`PreprocessSummary` is a tiny dataclass that the Flask template already knows
how to render (`summary.initial_rows`, …).
"""
from __future__ import annotations

from typing import Sequence
from collections import Counter
import os
from collections import Counter, OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (FunctionTransformer, PowerTransformer,
                                   QuantileTransformer, RobustScaler)
from .logging_config import setup_logger

# Ініціалізація логера
logger = setup_logger('preprocessing', 'preprocessing')

# ────────────────────────────────────────────────────────────
# 1.  CONSTANTS THAT DO NOT CHANGE INSIDE THE APP
# ────────────────────────────────────────────────────────────
DATE_MAP: Dict[str, str] = {
    "Monday": "2023-11-06 12:00:00",
    "Tuesday": "2023-11-07 12:00:00",
    "Wednesday": "2023-11-08 12:00:00",
    "Thursday-Morning": "2023-11-09 09:00:00",
    "Thursday-Afternoon": "2023-11-09 15:00:00",
    "Friday-Morning": "2023-11-10 09:00:00",
    "Friday-Afternoon1": "2023-11-10 13:00:00",
    "Friday-Afternoon2": "2023-11-10 17:00:00",
}

CATEGORY_LABELS: Dict[str, List[str]] = {
    "BENIGN": ["BENIGN"],
    "DoS": ["DDoS", "DoS slowloris", "DoS Hulk",
            "DoS GoldenEye", "DoS Slowhttptest"],
    "PortScan": ["PortScan"],
    "Bot_Infiltration": ["Bot", "Infiltration"],
    "Web": ["Web Attack – Brute Force",
            "Web Attack – XSS", "Web Attack – Sql Injection"],
    "FTP_SSH_Patator": ["FTP-Patator", "SSH-Patator"],
    "Heartbleed": ["Heartbleed"],
}

GROUP_FEATURES: Dict[str, List[str]] = {
    "dos": ['Fwd Packets/s', 'Bwd Packets/s', 'Flow Duration',
            'Flow IAT Min', 'Flow IAT Max', 'SYN Flag Count',
            'PSH Flag Count'],
    "portscan": ['SYN Flag Count', 'FIN Flag Count', 'RST Flag Count',
                 'Total Fwd Packets', 'Total Backward Packets'],
    "bot_infiltration": ['Flow Duration', 'Fwd IAT Std', 'Bwd IAT Std',
                         'Fwd PSH Flags', 'Bwd URG Flags', 'Down/Up Ratio'],
    "web": ['Fwd Header Length', 'Bwd Header Length',
            'Packet Length Variance', 'ACK Flag Count', 'Average Packet Size'],
    "ftp_ssh_patator": ['Fwd Avg Bytes/Bulk', 'Fwd Avg Packets/Bulk',
                        'Bwd Avg Bytes/Bulk', 'Active Mean', 'Idle Mean'],
    "heartbleed": ['Fwd Packet Length Max', 'Fwd Packet Length Min',
                   'Fwd IAT Min', 'Total Length of Fwd Packets',
                   'Packet Length Std'],
}

BASE_FEATURES: List[str] = [
    'Flow Bytes/s', 'Flow Packets/s', 'Average Packet Size', 'Down/Up Ratio',
    'Packet Length Mean', 'Packet Length Std', 'Min Packet Length',
    'Max Packet Length', 'Flow IAT Mean', 'Flow IAT Std', 'Fwd IAT Mean',
    'Bwd IAT Mean', 'SYN Flag Count', 'FIN Flag Count', 'RST Flag Count',
    'PSH Flag Count', 'ACK Flag Count', 'Active Mean', 'Idle Mean',
    'Subflow Fwd Packets', 'Subflow Bwd Packets', 'Label', 'dow', 'hour',
    'time',
]

STD_FEATURES: List[str] = [
    'Fwd Packet Length Std', 'Bwd Packet Length Std', 'Flow IAT Std',
    'Fwd IAT Std', 'Bwd IAT Std', 'Packet Length Std', 'Active Std',
    'Idle Std',
]

LOG1P_FEATURES = [
    'Active Mean', 'Average Packet Size', 'Bwd Header Length',
    'Bwd IAT Mean', 'Bwd IAT Std', 'Down/Up Ratio', 'Flow Bytes/s',
    'Flow Duration', 'Flow IAT Mean', 'Flow IAT Min', 'Flow IAT Std',
    'Flow Packets/s', 'Fwd Header Length', 'Fwd IAT Mean', 'Fwd IAT Std',
    'Fwd PSH Flags', 'Fwd Packets/s', 'Idle Mean', 'Max Packet Length',
    'Min Packet Length', 'Packet Length Mean', 'Packet Length Std',
    'Packet Length Variance',
]

YEO_JOHNSON_FEATURES = [
    'Total Length of Fwd Packets', 'Fwd Packet Length Max',
]

QUANTILE_FEATURES = [
    'Bwd Packets/s', 'Subflow Bwd Packets', 'Subflow Fwd Packets',
    'Total Backward Packets', 'Total Fwd Packets',
]

ROBUST_FEATURES = [
    'Bwd Avg Bytes/Bulk', 'Bwd URG Flags', 'Flow IAT Max',
    'Fwd Avg Bytes/Bulk', 'Fwd Avg Packets/Bulk', 'Fwd IAT Min',
    'Fwd Packet Length Min',
]

BIN_FEATURES = [
    'Flow IAT Min', 'Fwd Header Length', 'Fwd PSH Flags',
    'Packet Length Variance', 'Total Backward Packets',
    'Total Fwd Packets',
]

DROP_FEATURES = [
    'Packet Length Std', 'Max Packet Length', 'Average Packet Size',
    'Flow IAT Std', 'Fwd IAT Mean', 'Bwd IAT Mean', 'Fwd Header Length',
    'Bwd Header Length', 'Label', 'Category', 'dow', 'hour',
]

_TRANSFORMERS = {
    'log1p': FunctionTransformer(np.log1p, validate=False),
    'yeo_johnson': PowerTransformer(method='yeo-johnson', standardize=False),
    'quantile': QuantileTransformer(output_distribution='normal',
                                    random_state=0),
    'robust': RobustScaler(),
}

SMOTE_RANDOM_STATE: int = 1         # reproducibility for class balancing
TEST_SIZE: float = 0.15             # 15 % held-out test
VAL_SPLIT: float = 0.12             # 12 % of the remaining temp-set = 10 % overall
RANDOM_STATE: int = 42              # reproducibility for splits
# ────────────────────────────────────────────────────────────


@dataclass(slots=True)
class PreprocessSummary:
    """Just enough meta-data for the Bootstrap card in the template."""
    filename: str
    initial_rows: int
    after_clean_rows: int
    missing_before: int
    negatives_before: int
    processed_path: str
    df_head: str = ""  # HTML representation of dataframe head


# ════════════════════════════════════════════════════════════
# 2.  SMALL RE-USABLE STEPS
# ════════════════════════════════════════════════════════════
def _add_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    try:
        df['timestamp'] = pd.to_datetime(df['Day'].map(DATE_MAP))
        result = df.set_index('timestamp').drop(columns='Day')
        logger.info("Успішно додано часовий індекс")
        return result
    except Exception as e:
        logger.error(f"Помилка при додаванні часового індексу: {str(e)}")
        raise


def _engineer_time_features(df: pd.DataFrame) -> pd.DataFrame:
    try:
        df['dow'] = df.index.dayofweek
        df['hour'] = df.index.hour
        logger.info("Успішно додано часові ознаки")
        return df
    except Exception as e:
        logger.error(f"Помилка при додаванні часових ознак: {str(e)}")
        raise


def _clean_data(df: pd.DataFrame) -> pd.DataFrame:
    try:
        exclude = ['Init_Win_bytes_forward', 'Init_Win_bytes_backward', 'Label']
        numeric_cols = df.select_dtypes(include=[np.number]).columns.difference(exclude)

        initial_rows = len(df)
        df[numeric_cols] = df[numeric_cols].mask(df[numeric_cols] < 0)
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        result = df.dropna(axis=0).reset_index(drop=True)
        
        logger.info(f"Очищено дані: {initial_rows - len(result)} рядків видалено")
        return result
    except Exception as e:
        logger.error(f"Помилка при очищенні даних: {str(e)}")
        raise


def _apply_transformers(df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
    try:
        feature_map = {
            'log1p': LOG1P_FEATURES,
            'yeo_johnson': YEO_JOHNSON_FEATURES,
            'quantile': QUANTILE_FEATURES,
            'robust': ROBUST_FEATURES,
        }
        df_out = df.copy()
        for key, feats in feature_map.items():
            active = [c for c in feats if c in df_out.columns]
            if not active:
                continue
            tf = _TRANSFORMERS[key]
            if fit:
                df_out.loc[:, active] = tf.fit_transform(df_out[active])
            else:
                df_out.loc[:, active] = tf.transform(df_out[active])
        logger.info("Успішно застосовано трансформації до ознак")
        return df_out
    except Exception as e:
        logger.error(f"Помилка при застосуванні трансформацій: {str(e)}")
        raise


def _bin_features(df: pd.DataFrame, n_bins: int = 5) -> pd.DataFrame:
    try:
        df_out = df.copy()
        for feat in BIN_FEATURES:
            if feat in df_out.columns:
                df_out[f"{feat}_bin"] = pd.qcut(df_out[feat], q=n_bins,
                                                labels=False, duplicates="drop")
        logger.info(f"Успішно розбито ознаки на {n_bins} бінів")
        return df_out
    except Exception as e:
        logger.error(f"Помилка при розбитті ознак на біни: {str(e)}")
        raise


def _map_labels(df: pd.DataFrame) -> pd.DataFrame:
    try:
        df = df.copy()
        # ——— НОВАЯ НОРМАЛИЗАЦИЯ ———
        import unicodedata, re
        df['Label'] = df['Label'].apply(lambda s: unicodedata.normalize('NFKC', s))
        df['Label'] = (
            df['Label']
            .str.replace(r'[\u2010\u2011\u2012\u2013\u2014\u2015\u2212\uFFFD]', '-', regex=True)
            .str.strip()
        )
        # ——————————————

        # Заменяем ASCII-дефис на en-dash, как в вашем CATEGORY_LABELS
        df['Label'] = df['Label'].str.replace('-', '–', regex=False)

        inverted = {raw: cat
                    for cat, raws in CATEGORY_LABELS.items()
                    for raw in raws}
        df['Category'] = df['Label'].map(inverted)

        missing = df['Category'].isna()
        if missing.any():
            unmapped = df.loc[missing, 'Label'].unique()
            logger.warning(f"Нераспознанные метки: {unmapped}")

        code_map = {cat: i for i, cat in enumerate(CATEGORY_LABELS)}
        df['label_code'] = df['Category'].map(code_map)

        before = len(df)
        df = df[df['label_code'].notna()].copy()
        df['label_code'] = df['label_code'].astype(int)
        removed = before - len(df)
        if removed:
            logger.info(f"Удалено {removed} строк с ненайденным label_code")

        logger.info("Успішно зіставлено мітки")
        return df
    except Exception as e:
        logger.error(f"Помилка при зіставленні міток: {str(e)}")
        raise


def _add_time_feature(df: pd.DataFrame) -> pd.DataFrame:
    try:
        df = df.copy()
        df['time'] = df['dow'] * 24 + df['hour']
        logger.info("Успішно додано часову ознаку")
        return df
    except Exception as e:
        logger.error(f"Помилка при додаванні часової ознаки: {str(e)}")
        raise


def _drop_and_select(df: pd.DataFrame) -> pd.DataFrame:
    try:
        df = df.drop(columns=DROP_FEATURES, errors='ignore').copy()

        group_cols = [c for cols in GROUP_FEATURES.values() for c in cols]
        wanted = list(OrderedDict.fromkeys(
            BASE_FEATURES + STD_FEATURES + group_cols
        ))
        present = [c for c in wanted if c in df.columns]

        for extra in ('label_code', 'composite'):
            if extra in df.columns and extra not in present:
                present.append(extra)

        result = df[present]
        logger.info(f"Відібрано {len(present)} ознак")
        return result
    except Exception as e:
        logger.error(f"Помилка при відборі ознак: {str(e)}")
        raise


def _split_and_balance(
    df: pd.DataFrame,
    *,
    test_size: float = TEST_SIZE,
    val_split: float = VAL_SPLIT,
    random_state: int = RANDOM_STATE,
    smote_state: int = SMOTE_RANDOM_STATE,
) -> Dict[str, pd.DataFrame]:
    """
    Розбиває на train / val / test із часовою стратифікацією
    та балансує train через SMOTE.
    Повертає той самий набір, що й оригінальний скрипт.
    """
    # ── 0. розділяємо ціль і "composite" (час × клас)
    X = df.drop(columns=['label_code', 'composite'])
    y = df['label_code']
    c = df['composite']

    # ── 1. hold-out test
    X_tmp, X_test, y_tmp, y_test, c_tmp, c_test = train_test_split(
        X, y, c,
        test_size=test_size,
        stratify=c,
        random_state=random_state,
    )
    # ── 2. train / val з решти
    X_train, X_val, y_train, y_val, c_train, c_val = train_test_split(
        X_tmp, y_tmp, c_tmp,
        test_size=val_split,
        stratify=c_tmp,
        random_state=random_state,
    )

    # ── 3. SMOTE на train
    min_count = y_train.value_counts().min()
    k = max(1, min(min_count - 1, 5))          # адаптивно, щоб не впасти
    sm = SMOTE(k_neighbors=k, random_state=smote_state)
    X_train_bal, y_train_bal = sm.fit_resample(X_train, y_train)

    # (не обов'язково, але корисно подивитись у консолі)
    print("TRAIN classes:", Counter(y_train_bal))
    print("VAL   classes:", Counter(y_val))
    print("TEST  classes:", Counter(y_test))

    # ── 4. повертаємо так само, як у старому pipeline
    return {
        'X_train_bal': X_train_bal,
        'y_train_bal': y_train_bal,
        'X_val':       X_val,
        'y_val':       y_val,
        'X_test':      X_test,
        'y_test':      y_test,
    }


# ════════════════════════════════════════════════════════════
# 3.  PUBLIC HELPERS FOR FLASK
# ════════════════════════════════════════════════════════════
def list_raw_files(raw_dir: str | os.PathLike) -> List[str]:
    try:
        files = [f for f in os.listdir(raw_dir) if f.endswith('.csv')]
        logger.info(f"Знайдено {len(files)} CSV файлів у директорії {raw_dir}")
        return files
    except Exception as e:
        logger.error(f"Помилка при пошуку CSV файлів: {str(e)}")
        return []

def preprocess_files_combined(
    *,
    filenames: Sequence[str],
    raw_dir: str | os.PathLike,
    processed_dir: str | os.PathLike,
) -> list[PreprocessSummary]:
    """Preprocess several CSV files as one combined dataset."""
    try:
        summaries: list[PreprocessSummary] = []
        dfs: list[pd.DataFrame] = []

        # ── 1. read all files and collect meta-info
        for filename in filenames:
            logger.info(f"Початок обробки файлу {filename}")
            path = Path(raw_dir) / filename
            if not path.exists():
                raise FileNotFoundError(path)

            df_raw = pd.read_csv(path)
            df_raw.columns = df_raw.columns.str.strip()
            df_raw['Label'] = df_raw['Label'].str.replace('-', '–', regex=False)
            df_raw['Day'] = path.stem  # потрібно для DATE_MAP

            initial_rows = len(df_raw)
            missing_before = int(df_raw.isna().sum().sum())
            numeric_cols = df_raw.select_dtypes(include=[np.number]).columns
            negatives_before = int((df_raw[numeric_cols] < 0).sum().sum())

            summaries.append(
                PreprocessSummary(
                    filename=filename,
                    initial_rows=initial_rows,
                    after_clean_rows=0,  # заповнимо пізніше
                    missing_before=missing_before,
                    negatives_before=negatives_before,
                    processed_path="",  # ← заповнимо після save
                    df_head="",          # ← заповнимо після обробки
                )
            )

            dfs.append(df_raw)

        if not dfs:
            return []

        # ── 2. combine all dataframes
        df_all = pd.concat(dfs, ignore_index=True)

        df = _add_datetime_index(df_all)
        df = _engineer_time_features(df)
        df = _clean_data(df)

        clean_len = len(df)
        for s in summaries:
            s.after_clean_rows = clean_len

        df = _apply_transformers(df)
        df = _bin_features(df)
        df = _map_labels(df)
        df = _add_time_feature(df)
        df['composite'] = df['label_code'] * 168 + df['time']
        df_final = _drop_and_select(df)

        df_head_html = df_final.head().to_html(classes="table table-sm", index=False)
        for s in summaries:
            s.df_head = df_head_html
            s.processed_path = str(Path(processed_dir).resolve())

        # ── 3. train / val / test + SMOTE on combined dataset
        splits = _split_and_balance(df_final)

        train = splits['X_train_bal'].copy()
        train['label_code'] = splits['y_train_bal']
        val = splits['X_val'].copy()
        val['label_code'] = splits['y_val']
        test = splits['X_test'].copy()
        test['label_code'] = splits['y_test']

        combo_stem = "+".join(Path(f).stem for f in filenames)
        for name, frame in [('train', train), ('val', val), ('test', test)]:
            if isinstance(frame, pd.Series):
                frame = frame.to_frame(name='label_code')
            out_path = Path(processed_dir) / f"{combo_stem}_{name}.csv"
            frame.to_csv(out_path, index=False)
            print(f"Saved {name:<5} → {out_path}")

        logger.info(f"Успішно оброблено {len(filenames)} файлів у комбінації {combo_stem}")
        return summaries
    except Exception as e:
        logger.error(f"Помилка при обробці файлів: {str(e)}")
        raise
