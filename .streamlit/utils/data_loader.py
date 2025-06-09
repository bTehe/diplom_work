# utils/data_loader.py
"""
Модуль для завантаження та попереднього перегляду даних із CSV-файлів.

Сутності модуля:
- logger: об'єкт logging.Logger, налаштований для запису повідомлень, пов’язаних із завантаженням даних.
- load_csv_preview: функція для зчитування перших n рядків CSV-файлу та повернення їх у вигляді HTML-таблиці.
"""

import logging

import pandas as pd

from .logging_config import setup_logger

# Ініціалізація логера для модуля
logger: logging.Logger = setup_logger(__name__, "data_loader.log")


def load_csv_preview(path: str, n: int = 5) -> str:
    """
    Зчитує перші `n` рядків CSV-файлу й повертає їх у вигляді HTML-таблиці.

    Використовує Bootstrap-класи для форматування таблиці:
      - `table`
      - `table-sm`
      - `table-striped`

    :param path: Шлях до CSV-файлу.
    :param n: Кількість рядків для попереднього перегляду (за замовчуванням 5).
    :return: HTML-рядок із таблицею або HTML-блок із повідомленням про помилку.
    """
    try:
        logger.info(f"Спроба завантаження CSV-файлу: {path}")
        df: pd.DataFrame = pd.read_csv(path, nrows=n)
        logger.info(f"Успішно завантажено {n} рядків із файлу: {path}")

        html: str = df.to_html(
            classes="table table-sm table-striped",
            index=False
        )
        return html

    except Exception as error:
        logger.error(f"Помилка під час читання файлу {path}: {error}")
        return (
            "<div class='alert alert-danger'>"
            f"Помилка при завантаженні превʼю: {error}"
            "</div>"
        )


__all__ = ["load_csv_preview"]
