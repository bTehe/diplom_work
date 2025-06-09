from pathlib import Path
from collections import OrderedDict, Counter
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (FunctionTransformer, PowerTransformer,
                                   QuantileTransformer, RobustScaler)

# Конфігураційні константи
DATA_DIR = Path("../Dataset")
OUTPUT_DIR = Path("../Filtered datasets")

# мапування днів неділі в timestamp
DATE_MAP: Dict[str, str] = {
    'Monday': '2023-11-06 12:00:00',
    'Tuesday': '2023-11-07 12:00:00',
    'Wednesday': '2023-11-08 12:00:00',
    'Thursday-Morning': '2023-11-09 09:00:00',
    'Thursday-Afternoon': '2023-11-09 15:00:00',
    'Friday-Morning': '2023-11-10 09:00:00',
    'Friday-Afternoon1': '2023-11-10 13:00:00',
    'Friday-Afternoon2': '2023-11-10 17:00:00',
}

CATEGORY_LABELS: Dict[str, List[str]] = {
    'BENIGN': ['BENIGN'],
    'DoS': ['DDoS', 'DoS slowloris', 'DoS Hulk', 'DoS GoldenEye', 'DoS Slowhttptest'],
    'PortScan': ['PortScan'],
    'Bot_Infiltration': ['Bot', 'Infiltration'],
    'Web': ['Web Attack – Brute Force', 'Web Attack – XSS', 'Web Attack – Sql Injection'],
    'FTP_SSH_Patator': ['FTP-Patator', 'SSH-Patator'],
    'Heartbleed': ['Heartbleed'],
}

GROUP_FEATURES: Dict[str, List[str]] = {
    'dos': ['Fwd Packets/s', 'Bwd Packets/s', 'Flow Duration', 'Flow IAT Min', 'Flow IAT Max', 'SYN Flag Count', 'PSH Flag Count'],
    'portscan': ['SYN Flag Count', 'FIN Flag Count', 'RST Flag Count', 'Total Fwd Packets', 'Total Backward Packets'],
    'bot_infiltration': ['Flow Duration', 'Fwd IAT Std', 'Bwd IAT Std', 'Fwd PSH Flags', 'Bwd URG Flags', 'Down/Up Ratio'],
    'web': ['Fwd Header Length', 'Bwd Header Length', 'Packet Length Variance', 'ACK Flag Count', 'Average Packet Size'],
    'ftp_ssh_patator': ['Fwd Avg Bytes/Bulk', 'Fwd Avg Packets/Bulk', 'Bwd Avg Bytes/Bulk', 'Active Mean', 'Idle Mean'],
    'heartbleed': ['Fwd Packet Length Max', 'Fwd Packet Length Min', 'Fwd IAT Min', 'Total Length of Fwd Packets', 'Packet Length Std'],
}

BASE_FEATURES: List[str] = [
    'Flow Bytes/s', 'Flow Packets/s', 'Average Packet Size', 'Down/Up Ratio',
    'Packet Length Mean', 'Packet Length Std', 'Min Packet Length', 'Max Packet Length',
    'Flow IAT Mean', 'Flow IAT Std', 'Fwd IAT Mean', 'Bwd IAT Mean',
    'SYN Flag Count', 'FIN Flag Count', 'RST Flag Count', 'PSH Flag Count', 'ACK Flag Count',
    'Active Mean', 'Idle Mean', 'Subflow Fwd Packets', 'Subflow Bwd Packets',
    'Label', 'dow', 'hour', 'time'
]

STD_FEATURES: List[str] = [
    'Fwd Packet Length Std', 'Bwd Packet Length Std', 'Flow IAT Std',
    'Fwd IAT Std', 'Bwd IAT Std', 'Packet Length Std', 'Active Std', 'Idle Std'
]
def load_and_concat_csvs(data_dir: Path) -> pd.DataFrame:
    """
    Завантажує всі CSV-файли з директорії, додає стовпець 'Day' на основі імені файлу
    та об'єднує їх в один DataFrame.

    :param data_dir: шлях до директорії з CSV-файлами
    :return: конкатенований DataFrame з сирими даними
    """
    csv_paths = sorted(data_dir.glob("*.csv"))
    dfs: List[pd.DataFrame] = []

    for path in csv_paths:
        day_label = path.stem  # мітка дня з імені файлу
        df_temp = pd.read_csv(path)
        df_temp['Day'] = day_label
        dfs.append(df_temp)

    concatenated = pd.concat(dfs, ignore_index=True)
    concatenated.columns = concatenated.columns.str.strip()  # очищення пробілів в назвах стовпців
    return concatenated
def add_datetime_index(df: pd.DataFrame, date_map: Dict[str, str]) -> pd.DataFrame:
    """
    Перетворює стовпець 'Day' у datetime-індекс на основі мапи дат,
    встановлює його як індекс і видаляє стовпець 'Day'.

    :param df: DataFrame з колонкою 'Day'
    :param date_map: словник мітка -> datetime рядок
    :return: DataFrame з datetime-індексом
    """
    df['timestamp'] = pd.to_datetime(df['Day'].map(date_map))
    df = df.set_index('timestamp').drop(columns=['Day'])
    return df
def engineer_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Додає циклічні ознаки для дня тижня та години (синус/косинус).

    :param df: DataFrame з datetime-індексом
    :return: DataFrame з новими часовими ознаками
    """
    df['dow'] = df.index.dayofweek  # день тижня (0=Понеділок)
    df['hour'] = df.index.hour  # година доби
    return df
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Очищення даних:
      - Негативні числа у числових колонках замінюються на NaN
      - Нескінченності замінюються на NaN
      - Видалення рядків з будь-якими NaN
      - Скидання індексу

    :param df: початковий або частково оброблений DataFrame
    :return: очищений DataFrame, готовий до аналізу або моделювання
    """
    # Визначення числових колонок (без деяких виключень)
    exclude = ['Init_Win_bytes_forward', 'Init_Win_bytes_backward', 'Label']
    numeric_cols = df.select_dtypes(include=[np.number]).columns.difference(exclude)

    # Маска негативних значень та нескінченностей
    df[numeric_cols] = df[numeric_cols].mask(df[numeric_cols] < 0)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Видалення рядків з пропусками
    df = df.dropna(axis=0, how='any')

    # Скидання індексу
    df = df.reset_index(drop=True)
    return df
raw_df = load_and_concat_csvs(DATA_DIR)
df_indexed = add_datetime_index(raw_df, DATE_MAP)
df_features = engineer_time_features(df_indexed)
df_clean = clean_data(df_features)
# 1) log1p: сильна права асиметрія (skew > 1)
LOG1P_FEATURES = [
    'Active Mean',
    'Average Packet Size',
    'Bwd Header Length',
    'Bwd IAT Mean',
    'Bwd IAT Std',
    'Down/Up Ratio',
    'Flow Bytes/s',
    'Flow Duration',
    'Flow IAT Mean',
    'Flow IAT Min',
    'Flow IAT Std',
    'Flow Packets/s',
    'Fwd Header Length',
    'Fwd IAT Mean',
    'Fwd IAT Std',
    'Fwd PSH Flags',
    'Fwd Packets/s',
    'Idle Mean',
    'Max Packet Length',
    'Min Packet Length',
    'Packet Length Mean',
    'Packet Length Std',
    'Packet Length Variance'
]

# 2) Yeo–Johnson: сильна ліва асиметрія (skew < –1), помірна права асиметрія (0.5 < skew ≤ 1), або наявність нулів/від’ємних
YEO_JOHNSON_FEATURES = [
    'Total Length of Fwd Packets'
    'Fwd Packet Length Max'
]

# 3) QuantileTransformer: надзвичайно важкі хвости (skew > 50)
QUANTILE_FEATURES = [
    'Bwd Packets/s',
    'Subflow Bwd Packets',
    'Subflow Fwd Packets',
    'Total Backward Packets',
    'Total Fwd Packets'
]

# 4) За рештою ознак (|skew| ≤ 0.5) достатньо RobustScaler або залишити без трансформації:
ROBUST_FEATURES = [
    'Bwd Avg Bytes/Bulk',
    'Bwd URG Flags',
    'Flow IAT Max',
    'Fwd Avg Bytes/Bulk',
    'Fwd Avg Packets/Bulk',
    'Fwd IAT Min',
    'Fwd Packet Length Min',
]


BIN_FEATURES: List[str] = [
    'Flow IAT Min', 'Fwd Header Length', 'Fwd PSH Flags',
    'Packet Length Variance', 'Total Backward Packets', 'Total Fwd Packets'
]

DROP_FEATURES: List[str] = [
    'Packet Length Std', 'Max Packet Length', 'Average Packet Size',
    'Flow IAT Std', 'Fwd IAT Mean', 'Bwd IAT Mean',
    'Fwd Header Length', 'Bwd Header Length', 'Label', 'Category', 'dow','hour'
]

_transformers: Dict[str, object] = {
    'log1p': FunctionTransformer(np.log1p, validate=False),
    'yeo_johnson': PowerTransformer(method='yeo-johnson', standardize=False),
    'quantile': QuantileTransformer(output_distribution='normal',
                                    random_state=0),
    'robust': RobustScaler(),
}
def apply_transformers(df: pd.DataFrame,
                       feature_map: Dict[str, List[str]],
                       transformers: Dict[str, object],
                       fit: bool = True) -> pd.DataFrame:
    """
    Застосовує набір трансформерів до відповідних списків ознак.

    Параметри
    ----------
    df : pandas.DataFrame
        Вхідний DataFrame.
    feature_map : dict
        Відображення ключів трансформерів на списки імен ознак.
    transformers : dict
        Відображення ключів на інстанси scikit-learn трансформерів.
    fit : bool, default=True
        Якщо True — спочатку навчає трансформери на даних, інакше — лише трансформує.

    Повертає
    -------
    pandas.DataFrame
        DataFrame з трансформованими ознаками.
    """
    df_out = df.copy()
    for key, features in feature_map.items():
        transformer = transformers[key]
        present = [f for f in features if f in df_out.columns]
        if not present:
            continue
        data = df_out[present]
        if fit:
            df_out.loc[:, present] = transformer.fit_transform(data)
        else:
            df_out.loc[:, present] = transformer.transform(data)
    return df_out
def bin_features(df: pd.DataFrame,
                 features: List[str],
                 n_bins: int = 5) -> pd.DataFrame:
    """
    Квантільно бінує вибрані неперервні ознаки.

    Параметри
    ----------
    df : pandas.DataFrame
        Вхідний DataFrame.
    features : list of str
        Список імен ознак для бінування.
    n_bins : int, default=5
        Кількість бінів (квантилів).

    Повертає
    -------
    pandas.DataFrame
        Копія DataFrame з новими стовпцями '{feature}_bin'.
    """
    df_out = df.copy()
    for feat in features:
        if feat in df_out.columns:
            df_out[f'{feat}_bin'] = pd.qcut(
                df_out[feat], q=n_bins, labels=False, duplicates='drop'
            )
    return df_out
def map_labels(df: pd.DataFrame,
               category_labels: Dict[str, List[str]]) -> pd.DataFrame:
    """
    Відображає сирі рядки міток у числові коди категорій.

    Параметри
    ----------
    df : pandas.DataFrame
        DataFrame із стовпцем 'Label'.
    category_labels : dict
        Відображення категорій на списки сирих міток.

    Повертає
    -------
    pandas.DataFrame
        DataFrame з доданими стовпцями 'Category' та 'label_code'.
    """
    
    df_out = df.copy()
    # Clean unicode replacements
    df_out['Label'] = df_out['Label'].str.replace('�', '–', regex=False)

    # Invert mapping for lookup
    label_to_cat = {
        raw: cat
        for cat, raws in category_labels.items()
        for raw in raws
    }
    df_out['Category'] = df_out['Label'].map(label_to_cat)

    # Numeric codes for each high-level category
    cats = list(category_labels.keys())
    code_map = {cat: idx for idx, cat in enumerate(cats)}
    df_out['label_code'] = df_out['Category'].map(code_map)
    return df_out
def add_time_feature(df: pd.DataFrame) -> pd.DataFrame:
    """
    Комбінує день тижня та годину в один індекс часу.

    Параметри
    ----------
    df : pandas.DataFrame
        DataFrame зі стовпцями 'dow' та 'hour'.

    Повертає
    -------
    pandas.DataFrame
        DataFrame з новим стовпцем 'time'.
    """
    
    df_out = df.copy()
    df_out['time'] = df_out['dow'] * 24 + df_out['hour']
    return df_out
def drop_and_select(df: pd.DataFrame,
                    drop_cols: List[str],
                    group_features: Dict[str, List[str]],
                    base_features: List[str],
                    std_features: List[str]) -> pd.DataFrame:
    """
    Видаляє небажані стовпці та вибирає лише існуючі в заданому порядку.

    Параметри
    ----------
    df : pandas.DataFrame
        Вхідний DataFrame.
    drop_cols : list of str
        Список стовпців для видалення.
    group_features : dict
        Групи ознак для агрегації.
    base_features : list of str
        Базовий список ознак.
    std_features : list of str
        Додатковий список ознак.

    Повертає
    -------
    pandas.DataFrame
        Відфільтрований DataFrame.
    """
    df_out = df.drop(columns=drop_cols, errors='ignore').copy()
    # Flatten group columns and preserve order without duplicates
    group_cols = [c for cols in group_features.values() for c in cols]
    all_feats = list(OrderedDict.fromkeys(base_features + std_features + group_cols))
    existing = [c for c in all_feats if c in df_out.columns]
    # Keep label_code and composite if present
    for extra in ['label_code', 'composite']:
        if extra in df_out.columns:
            existing.append(extra)
    return df_out[existing]
def split_and_balance(df: pd.DataFrame,
                      test_size: float,
                      val_split: float,
                      random_state: int,
                      smote_state: int) -> Dict[str, pd.DataFrame]:
    """
    Розбиває дані на тренувальний, валідаційний та тестовий набори,
    стратифікує за часовим композитом та застосовує SMOTE-балансування.

    Параметри
    ----------
    df : pandas.DataFrame
        DataFrame з 'label_code' та 'composite'.
    test_size : float
        Частка для тестового набору.
    val_split : float
        Частка від залишку для валідації.
    random_state : int
        Насіння для відтворюваності.
    smote_state : int
        Насіння SMOTE.

    Повертає
    -------
    dict of pandas.DataFrame
        Словник із ключами 'X_train_bal', 'y_train_bal', 'X_val', 'y_val',
        'X_test', 'y_test'.
    """
    X = df.drop(columns=['label_code', 'composite'])
    y = df['label_code']
    c = df['composite']

    # First split off test
    X_temp, X_test, y_temp, y_test, c_temp, c_test = train_test_split(
        X, y, c, test_size=test_size,
        stratify=c, random_state=random_state
    )
    # Then split temp into train/val
    X_train, X_val, y_train, y_val, c_train, c_val = train_test_split(
        X_temp, y_temp, c_temp, test_size=val_split,
        stratify=c_temp, random_state=random_state
    )

    # Determine SMOTE k_neighbors
    min_count = y_train.value_counts().min()
    k = max(1, min(min_count - 1, 5))
    print(f"SMOTE will use k_neighbors={k} (min class count = {min_count})")

    sm = SMOTE(k_neighbors=k, random_state=smote_state)
    X_train_bal, y_train_bal = sm.fit_resample(X_train, y_train)

    # Debug prints
    print("TRAIN classes:", Counter(y_train_bal))
    print("VAL   classes:", Counter(y_val))
    print("TEST  classes:", Counter(y_test))
    print("TRAIN times:", Counter(c_train % 168))
    print("VAL   times:", Counter(c_val % 168))
    print("TEST  times:", Counter(c_test % 168))

    return {
        'X_train_bal': X_train_bal, 'y_train_bal': y_train_bal,
        'X_val': X_val,          'y_val': y_val,
        'X_test': X_test,        'y_test': y_test,
    }

def export_datasets(datasets: Dict[str, pd.DataFrame],
                    base_path: str) -> None:
    """
    Зберігає набори даних у CSV-файли під вказаним базовим шляхом.

    Параметри
    ----------
    datasets : dict
        Словник з іменами наборів даних та DataFrame.
    base_path : str
        Базовий шлях для збереження файлів.

    Повертає
    -------
    None
    """

    for name, df in datasets.items():
        filename = f"{base_path}/{name}.csv"
        df.to_csv(filename, index=False)
        print(f"Saved {name} to {filename}")

SMOTE_RANDOM_STATE = 1
TEST_SIZE = 0.15
VAL_SPLIT = 0.12
RANDOM_STATE = 42

df_transformed = apply_transformers(
    df_clean,
    feature_map={
        'log1p': LOG1P_FEATURES,
        'yeo_johnson': YEO_JOHNSON_FEATURES,
        'quantile': QUANTILE_FEATURES,
        'robust': ROBUST_FEATURES,
    },
    transformers=_transformers,
    fit=True
)

df_binned = bin_features(df_transformed, BIN_FEATURES)
df_mapped = map_labels(df_binned, CATEGORY_LABELS)
df_time = add_time_feature(df_mapped)
df_time['composite'] = df_time['label_code'] * 168 + df_time['time']
df_final = drop_and_select(
    df_time, DROP_FEATURES, GROUP_FEATURES, BASE_FEATURES, STD_FEATURES
)

data_dict = split_and_balance(
    df_final, TEST_SIZE, VAL_SPLIT, RANDOM_STATE, SMOTE_RANDOM_STATE
)
export_datasets(data_dict, '../NN Datasets')