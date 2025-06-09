# app/utils/monitoring.py
"""
Модуль для моніторингу та обробки результатів інференсу інцидентів.

Сутності модуля:
- FILE_PATTERN_SUFFIX (str): суфікс імені файлів результатів.
- logger (logging.Logger): логер для запису подій модуля.
- list_incident_files(inc_dir): отримати список файлів результатів.
- load_incidents(file_name, inc_dir): завантажити один файл інцидентів у DataFrame.
- filter_incidents(df, pred_class, min_confidence): відфільтрувати інциденти.
- get_incident_statistics(df): розрахувати базову статистику по інцидентах.
- get_anomaly_incidents(df, confidence_threshold): знайти потенційні аномалії.
- export_incidents(df, format): експортувати DataFrame інцидентів у вказаний формат.
"""

import logging
from pathlib import Path

import pandas as pd

from .logging_config import setup_logger

# Суфікс файлів із результатами інференсу
FILE_PATTERN_SUFFIX: str = "_results.csv"

# Ініціалізація логера для модуля
logger: logging.Logger = setup_logger(__name__, "monitoring")


def list_incident_files(inc_dir: str | Path) -> list[str]:
    """
    Перелік CSV-файлів із результатами інцидентів у вказаній директорії.

    :param inc_dir: Шлях до директорії з файлами.
    :return: Список імен файлів, що закінчуються на '_results.csv'.
    """
    try:
        directory = Path(inc_dir)
        files = sorted(directory.glob(f"*{FILE_PATTERN_SUFFIX}"))
        filenames = [file.name for file in files]
        logger.info(
            f"Знайдено {len(filenames)} файлів із результатами "
            f"в директорії {directory}"
        )
        return filenames
    except Exception as error:
        logger.error(f"Помилка при пошуку файлів результатів: {error}")
        return []


def load_incidents(file_name: str, inc_dir: str | Path) -> pd.DataFrame:
    """
    Завантажує один файл результатів інференсу в DataFrame.

    :param file_name: Ім'я файлу з результатами.
    :param inc_dir: Шлях до директорії з файлами.
    :return: DataFrame із завантаженими даними або порожній DataFrame у разі помилки.
    """
    try:
        path = Path(inc_dir) / file_name
        logger.info(f"Завантаження файлу результатів: {file_name}")
        df = pd.read_csv(path)
        logger.info(f"Успішно завантажено {len(df)} рядків з файлу {file_name}")
        return df
    except Exception as error:
        logger.error(f"Помилка при завантаженні файлу {file_name}: {error}")
        return pd.DataFrame()


def filter_incidents(
    df: pd.DataFrame,
    pred_class: int | None = None,
    min_confidence: float = 0.0
) -> pd.DataFrame:
    """
    Фільтрує інциденти за передбаченим класом та/або мінімальною довірою.

    :param df: Вхідний DataFrame інцидентів.
    :param pred_class: Клас передбачення для фільтрації (необов’язково).
    :param min_confidence: Мінімальне значення довіри (за замовчуванням 0.0).
    :return: Відфільтрований DataFrame.
    """
    try:
        original_count = len(df)
        if pred_class is not None:
            df = df[df["pred"] == pred_class]
            filtered_count = original_count - len(df)
            logger.info(f"Відфільтровано за класом {pred_class}: {filtered_count} рядків")

        if min_confidence > 0.0:
            before = len(df)
            df = df[df["confidence"] >= min_confidence]
            filtered_count = before - len(df)
            logger.info(
                f"Відфільтровано за мінімальною довірою {min_confidence}: "
                f"{filtered_count} рядків"
            )

        return df
    except Exception as error:
        logger.error(f"Помилка при фільтрації інцидентів: {error}")
        return df


def get_incident_statistics(df: pd.DataFrame) -> dict:
    """
    Розраховує базову статистику по інцидентах.

    :param df: DataFrame інцидентів з колонками 'pred' та 'confidence'.
    :return: Словник зі статистикою:
        - total_incidents: загальна кількість інцидентів
        - avg_confidence: середня довіра
        - min_confidence: мінімальна довіра
        - max_confidence: максимальна довіра
        - class_distribution: розподіл інцидентів за класами
    """
    try:
        stats = {
            "total_incidents": len(df),
            "avg_confidence": float(df["confidence"].mean()) if not df.empty else 0.0,
            "min_confidence": float(df["confidence"].min()) if not df.empty else 0.0,
            "max_confidence": float(df["confidence"].max()) if not df.empty else 0.0,
            "class_distribution": df["pred"].value_counts().to_dict(),
        }
        logger.info(f"Розраховано статистику для {stats['total_incidents']} інцидентів")
        return stats
    except Exception as error:
        logger.error(f"Помилка при розрахунку статистики: {error}")
        return {}


def get_anomaly_incidents(
    df: pd.DataFrame,
    confidence_threshold: float = 0.5
) -> pd.DataFrame:
    """
    Визначає потенційні аномалії на основі порогу довіри.

    :param df: DataFrame інцидентів.
    :param confidence_threshold: Поріг довіри для виявлення аномалій.
    :return: DataFrame з інцидентами, довіра яких менша за поріг, відсортований за зростанням.
    """
    try:
        anomalies = df[df["confidence"] < confidence_threshold].sort_values("confidence")
        logger.info(
            f"Знайдено {len(anomalies)} потенційних аномалій "
            f"з порогом довіри {confidence_threshold}"
        )
        return anomalies
    except Exception as error:
        logger.error(f"Помилка при пошуку аномалій: {error}")
        return pd.DataFrame()


def export_incidents(df: pd.DataFrame, format: str = "csv") -> str:
    """
    Експортує інциденти у вказаний формат файлу.

    :param df: DataFrame інцидентів.
    :param format: Формат експорту: 'csv' або 'excel' (за замовчуванням 'csv').
    :return: Ім'я згенерованого файлу або пустий рядок у разі помилки.
    """
    try:
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        if format == "excel":
            filename = f"incidents_export_{timestamp}.xlsx"
            df.to_excel(filename, index=False)
        else:
            filename = f"incidents_export_{timestamp}.csv"
            df.to_csv(filename, index=False)

        logger.info(
            f"Успішно експортовано {len(df)} інцидентів у формат {format}"
        )
        return filename
    except Exception as error:
        logger.error(f"Помилка при експорті інцидентів: {error}")
        return ""


__all__ = [
    "list_incident_files",
    "load_incidents",
    "filter_incidents",
    "get_incident_statistics",
    "get_anomaly_incidents",
    "export_incidents",
]
