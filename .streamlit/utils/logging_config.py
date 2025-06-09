# app/utils/logging_config.py
"""
Модуль для налаштування логування у додатку.

Сутності модуля:
- LOG_DIR (pathlib.Path): директорія для зберігання лог-файлів.
- DATE_FORMAT (str): формат відображення дати у записах логів.
- FORMATTER_TEMPLATE (str): шаблон форматування повідомлень логера.
- setup_logger: функція для створення та налаштування логера з файловим та консольним хендлерами.
"""

import logging
from logging import Logger, Formatter, FileHandler, StreamHandler
from pathlib import Path
from datetime import datetime

# --- Константи модуля ---
LOG_DIR: Path = Path(".streamlit") / "logs"
DATE_FORMAT: str = "%Y-%m-%d %H:%M:%S"
FORMATTER_TEMPLATE: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


def setup_logger(name: str, log_file: str) -> Logger:
    """
    Створює і повертає налаштований логер з файловим та консольним хендлерами.

    :param name: Ім'я логера (рекомендовано використовувати __name__ модуля).
    :param log_file: Базова назва лог-файлу (без дати та розширення).
    :return: Об'єкт logging.Logger із встановленим рівнем INFO.
    """
    # Переконатися, що директорія для логів існує
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    # Ініціалізація логера
    logger: Logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Форматер для обох хендлерів
    formatter: Formatter = Formatter(fmt=FORMATTER_TEMPLATE, datefmt=DATE_FORMAT)

    # Файловий хендлер: ім'я файлу з додаванням дати
    dated_filename = f"{log_file}_{datetime.now():%Y%m%d}.log"
    file_handler: FileHandler = FileHandler(LOG_DIR / dated_filename, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Консольний хендлер
    console_handler: StreamHandler = StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


__all__ = ["setup_logger"]
